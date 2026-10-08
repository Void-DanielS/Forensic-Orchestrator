"""WindowsArtifactAgent: the Windows artefacts the SQLite parsers cannot see.

Deterministic (no model, role ``text``). Every finding cites the exact registry value, event record or
bytes it rests on (``src/bridge/windows_bridge.py``), and the HeuristicVerificationEngine re-reads each one.

=====================  ===================================================================================
category               source
=====================  ===================================================================================
win.system_info        SOFTWARE ``Windows NT\\CurrentVersion`` (product, build, owner, install date)
win.computer_name      SYSTEM ``Control\\ComputerName``
win.timezone           SYSTEM ``Control\\TimeZoneInformation``
win.shutdown           SYSTEM ``Control\\Windows\\ShutdownTime``
win.network_interface  SYSTEM ``Tcpip\\Parameters\\Interfaces\\{GUID}`` (IP, DHCP server, domain)
win.usb_device         SYSTEM ``Enum\\USBSTOR`` / ``Enum\\USB`` (vendor, product, serial, first / last seen)
win.portable_device    SOFTWARE ``Windows Portable Devices\\Devices`` and ``EMDMgmt`` (volume labels)
win.installed_program  SOFTWARE ``Uninstall`` (both views)
win.user_profile       SOFTWARE ``ProfileList`` (SID to profile path) and SAM user names (no hashes)
win.network_profile    SOFTWARE ``NetworkList\\Profiles``
win.userassist         NTUSER ``UserAssist`` (program, run count, last run; ROT13 decoded)
win.recent_doc         NTUSER ``RecentDocs`` and Office ``File MRU``
win.typed_url          NTUSER ``Internet Explorer\\TypedURLs``
win.search_term        NTUSER ``WordWheelQuery`` (Explorer search box)
win.run_mru            NTUSER ``RunMRU``
win.mountpoint         NTUSER ``MountPoints2`` (volumes / shares the user opened)
win.prefetch           ``Windows\\Prefetch\\*.pf`` (executable, run count, last run)
win.lnk                shortcuts and ``Recent`` items (target path, volume, target times)
win.recycle_bin        ``$Recycle.Bin\\$I*`` (Vista+) and ``RECYCLER\\INFO2`` (XP): original path, deletion time
win.network_card       SOFTWARE ``NetworkCards`` (adapter descriptions)
win.last_logon         SOFTWARE ``Winlogon`` (DefaultUserName / DefaultDomainName)
ie.url                 IE ``index.dat`` history / cache records
win.event              EVTX records of investigative event IDs (logons, USB, installs, start / stop)
=====================  ===================================================================================

Credential material is never read: SAM hashes, LSA secrets and DPAPI blobs stay out (FDX B8).
"""
from __future__ import annotations

import codecs
import json
import re
import struct
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from src.agents.base_agent import BaseAgent
from src.bridge.windows_bridge import (IE_URL_RECORD, bytes_record, filetime_to_iso, find_string, parse_prefetch,
                                       render_registry_value)
from src.verification.models import Finding

HIVE_NAMES = {"system": "SYSTEM", "software": "SOFTWARE", "sam": "SAM", "ntuser.dat": "NTUSER", "usrclass.dat": "USRCLASS"}
MAX_EVENTS_PER_LOG = 1500
MAX_PER_KEY = 400

# channel / provider keyword -> event IDs worth reporting
EVENT_IDS = {
    "security": {4624: "logon", 4634: "logoff", 4647: "logoff (user initiated)", 4720: "user account created",
                 4726: "user account deleted", 4732: "user added to group", 4648: "explicit-credential logon",
                 4688: "process created", 1102: "audit log cleared", 4616: "system time changed"},
    "system": {6005: "event log started (boot)", 6006: "event log stopped (shutdown)", 6008: "unexpected shutdown",
               1074: "shutdown / restart requested", 7045: "service installed", 104: "event log cleared",
               20001: "device driver installed", 20003: "device service added", 12: "OS start", 13: "OS shutdown",
               6013: "uptime"},
    "application": {11707: "product installed (MSI)", 11724: "product removed (MSI)", 1033: "MSI install result",
                    1034: "MSI removal result"},
    "driverframeworks-usermode": {2003: "USB device connected", 2100: "USB device event", 2102: "USB device removed"},
    "partition": {1006: "storage device connected"},
    "terminalservices-localsessionmanager": {21: "session logon", 23: "session logoff", 24: "session disconnected",
                                             25: "session reconnected"},
    "powershell": {400: "PowerShell engine started", 4104: "PowerShell script block", 800: "PowerShell pipeline"},
    "windows defender": {1116: "malware detected", 1117: "malware action taken"},
    "bits-client": {59: "BITS transfer started", 60: "BITS transfer stopped"},
    "taskscheduler": {106: "scheduled task registered", 200: "scheduled task action started"},
    "wlan-autoconfig": {8001: "connected to wireless network"},
}
# channels where an event ID is only meaningful from one provider (Application 1033 is also Security-SPP)
EVENT_PROVIDER = {"application": "MsiInstaller"}
_LOGON_TYPES = {"2", "7", "10", "11"}       # interactive, unlock, remote interactive, cached


def rot13(s: str) -> str:
    return codecs.decode(s, "rot13")


def _hexbytes(v: Any) -> bytes:
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    try:
        return bytes.fromhex(str(v))
    except ValueError:
        return b""


def _text(v: Any) -> str:
    """Display form of a registry value for claims (REG_SZ regipy could not decode arrives as bytes)."""
    if isinstance(v, (bytes, bytearray)):
        return _utf16z(bytes(v)).strip()
    return render_registry_value(v)


def _utf16z(b: bytes) -> str:
    end = next((i for i in range(0, len(b) - 1, 2) if b[i:i + 2] == b"\x00\x00"), len(b) - len(b) % 2)
    return b[:end].decode("utf-16-le", "replace")


class WindowsArtifactAgent(BaseAgent):
    name = "WindowsArtifactAgent"
    role = "text"

    def __init__(self, bridge, verifier=None, manifest: dict | None = None, **kw):
        super().__init__(bridge, verifier, **kw)
        self.meta: dict[str, dict] = {}
        if manifest:
            for f in manifest.get("files", []):
                self.meta[re.sub(r"^fs/", "", f["path"])] = f

    # ------------------------------------------------------------------ discovery
    def logical(self, rel: str) -> str:
        """The path inside the original volume (``_longpaths`` names are resolved through the manifest)."""
        m = self.meta.get(rel, {})
        return (m.get("original_path") or rel).replace("\\", "/")

    def kind(self, rel: str) -> str | None:
        lp = self.logical(rel)
        low = "/" + lp.lower().lstrip("/")      # rules below match "/<dir>/" also at the volume root
        name = PurePosixPath(low).name
        if "/regback/" in low or "/system volume information/" in low or "_restore{" in low:
            return None              # backup copies of hives: older states, duplicate findings
        if name in ("system", "software", "sam") and low.endswith(f"config/{name}"):
            return "hive"
        if name in ("ntuser.dat", "usrclass.dat"):
            return "hive"
        if name.endswith(".evtx"):
            return "evtx"
        if name.endswith(".pf") and "/prefetch/" in low:
            return "prefetch"
        if name.endswith(".lnk"):
            return "lnk"
        if name == "index.dat":
            return "index_dat"
        if re.search(r"/\$recycle\.bin/[^/]+/\$i[^/]*$", low):
            return "recycle_i"
        if name == "info2" and "/recycler/" in low:
            return "info2"
        return None

    def discover(self) -> list[str]:
        paths = list(self.meta) if self.meta else [e.path for e in self.bridge.fs.walk()]
        return sorted(p for p in paths if self.kind(p))

    def hive_type(self, rel: str) -> str | None:
        return HIVE_NAMES.get(PurePosixPath(self.logical(rel).lower()).name)

    def profile_user(self, rel: str) -> str | None:
        parts = PurePosixPath(self.logical(rel)).parts
        for i, p in enumerate(parts):
            if p.lower() in ("users", "documents and settings") and i + 1 < len(parts):
                return parts[i + 1]
        return None

    # ------------------------------------------------------------------ dispatch
    def analyze_artifact(self, rel: str) -> list[Finding]:
        kind = self.kind(rel)
        if kind is None:
            return []
        try:
            out = getattr(self, f"_{kind}")(rel)
        except Exception as exc:  # noqa: BLE001 - damaged artefact
            self.tracer.emit("parser_skip", artifact=rel, reason=f"{kind}: {type(exc).__name__}: {exc}")
            return []
        self.tracer.emit("parser", artifact=rel, category=kind, parser=kind, records=len(out), truncated=False)
        return [f for f in out if f.citations]

    def _emit(self, rel: str, category: str, claim: str, records: list, **value: Any) -> Finding | None:
        recs = [r for r in records if r is not None]
        if not recs:
            return None
        return self.finding(rel, category, claim, value={**value, "logical_path": self.logical(rel)},
                            citations=[r.to_citation() for r in recs])

    # ------------------------------------------------------------------ registry
    def _hive(self, rel: str) -> list[Finding]:
        ht = self.hive_type(rel)
        fn = {"SYSTEM": self._system, "SOFTWARE": self._software, "SAM": self._sam, "NTUSER": self._ntuser}.get(ht or "")
        if fn is None:
            return []
        if self.bridge.registry.key(rel, "\\") is None:
            return []
        return [f for f in fn(rel) if f is not None]

    def _rec(self, rel: str, key: str, name: str | None = None, field: str | None = None):
        return self.bridge.registry.record(rel, key, name, field)

    def _system(self, rel: str) -> list[Finding | None]:
        reg, out = self.bridge.registry, []
        sets = reg.control_sets(rel)
        if not sets:
            return out
        cs = sets[0]
        k = f"{cs}\\Control\\ComputerName\\ComputerName"
        name = reg.values(rel, k).get("ComputerName")
        if name:
            out.append(self._emit(rel, "win.computer_name", f"Computer name: {name}",
                                  [self._rec(rel, k, "ComputerName")], computer_name=name, control_set=cs))
        k = f"{cs}\\Control\\TimeZoneInformation"
        tz = reg.values(rel, k)
        tzname = _text(tz.get("TimeZoneKeyName") or tz.get("StandardName") or "")
        if tzname:
            bias = tz.get("Bias")
            recs = [self._rec(rel, k, n) for n in ("TimeZoneKeyName", "StandardName", "Bias", "ActiveTimeBias") if n in tz]
            out.append(self._emit(rel, "win.timezone", f"Time zone: {tzname} (bias {bias} min)", recs,
                                  timezone=tzname, bias_minutes=bias, active_bias=tz.get("ActiveTimeBias")))
        k = f"{cs}\\Control\\Windows"
        raw = reg.values(rel, k).get("ShutdownTime")
        if raw:
            b = _hexbytes(raw)
            when = filetime_to_iso(struct.unpack("<Q", b[:8])[0]) if len(b) >= 8 else None
            if when:
                out.append(self._emit(rel, "win.shutdown", f"Last recorded shutdown: {when} UTC",
                                      [self._rec(rel, k, "ShutdownTime")], shutdown_utc=when))
        base = f"{cs}\\Services\\Tcpip\\Parameters\\Interfaces"
        for guid in reg.subkeys(rel, base)[:MAX_PER_KEY]:
            k = f"{base}\\{guid}"
            v = reg.values(rel, k)
            fields = [n for n in ("DhcpIPAddress", "IPAddress", "DhcpServer", "DhcpDomain", "Domain", "DhcpDefaultGateway",
                                  "DefaultGateway", "DhcpNameServer", "NameServer", "DhcpSubnetMask")
                      if render_registry_value(v.get(n)).strip("|0. ")]
            if not fields:
                continue
            ip = render_registry_value(v.get("DhcpIPAddress") or v.get("IPAddress"))
            out.append(self._emit(rel, "win.network_interface",
                                  f"Network interface {guid}: IP {ip or '-'}, DHCP server {v.get('DhcpServer', '-')}, "
                                  f"domain {v.get('DhcpDomain') or v.get('Domain') or '-'}",
                                  [self._rec(rel, k, n) for n in fields],
                                  interface=guid, **{n: render_registry_value(v.get(n)) for n in fields}))
        # USB mass storage: Enum\USBSTOR\<Disk&Ven_..&Prod_..&Rev_..>\<serial>
        base = f"{cs}\\Enum\\USBSTOR"
        for dev in reg.subkeys(rel, base)[:MAX_PER_KEY]:
            m = dict(re.findall(r"(Ven|Prod|Rev)_([^&]*)", dev))
            for serial in reg.subkeys(rel, f"{base}\\{dev}")[:MAX_PER_KEY]:
                k = f"{base}\\{dev}\\{serial}"
                friendly = reg.values(rel, k).get("FriendlyName")
                recs = [self._rec(rel, k, None, "key_last_write"), self._rec(rel, k, "FriendlyName")]
                last = reg.read(rel, k)
                out.append(self._emit(rel, "win.usb_device",
                                      f"USB storage {m.get('Ven', '?')} {m.get('Prod', '?')} serial {serial.split('&')[0]} "
                                      f"('{friendly or dev}'), key last written {last}",
                                      recs, device_class=dev, vendor=m.get("Ven"), product=m.get("Prod"),
                                      revision=m.get("Rev"), serial=serial.split("&")[0], friendly_name=friendly,
                                      key_last_write_utc=last))
        base = f"{cs}\\Enum\\USB"
        for vidpid in reg.subkeys(rel, base)[:MAX_PER_KEY]:
            if not vidpid.upper().startswith("VID_"):
                continue
            for serial in reg.subkeys(rel, f"{base}\\{vidpid}")[:MAX_PER_KEY]:
                k = f"{base}\\{vidpid}\\{serial}"
                v = reg.values(rel, k)
                if "&" in serial and not v.get("FriendlyName"):
                    continue          # composite child interfaces without identity
                desc = v.get("FriendlyName") or render_registry_value(v.get("DeviceDesc")).split(";")[-1]
                out.append(self._emit(rel, "win.usb_device",
                                      f"USB device {vidpid} serial {serial} ('{desc}'), key last written {reg.read(rel, k)}",
                                      [self._rec(rel, k, None, "key_last_write"),
                                       self._rec(rel, k, "FriendlyName" if v.get("FriendlyName") else "DeviceDesc")],
                                      vid_pid=vidpid, serial=serial, description=desc,
                                      key_last_write_utc=reg.read(rel, k)))
        k = "\\MountedDevices"
        for n, raw in list(reg.values(rel, k).items())[:MAX_PER_KEY]:
            b = _hexbytes(raw)
            text = _utf16z(b) if len(b) > 24 else ""
            if "USBSTOR" in text.upper() or "REMOVABLE" in text.upper():
                out.append(self._emit(rel, "win.usb_device", f"Mounted device {n} -> {text[:160]}",
                                      [self._rec(rel, k, n)], mount=n, device_path=text))
        return out

    def _software(self, rel: str) -> list[Finding | None]:
        reg, out = self.bridge.registry, []
        k = "\\Microsoft\\Windows NT\\CurrentVersion"
        v = reg.values(rel, k)
        if v.get("ProductName"):
            fields = [n for n in ("ProductName", "CSDVersion", "CurrentVersion", "CurrentBuild", "CurrentBuildNumber",
                                  "RegisteredOwner", "RegisteredOrganization", "InstallDate", "SystemRoot",
                                  "EditionID", "ReleaseId", "DisplayVersion") if v.get(n) not in (None, "")]
            install = v.get("InstallDate")
            from datetime import datetime, timezone
            install_utc = datetime.fromtimestamp(int(install), timezone.utc).isoformat() if isinstance(install, int) and install else None
            out.append(self._emit(rel, "win.system_info",
                                  f"OS {v.get('ProductName')} {v.get('CSDVersion', '')} build "
                                  f"{v.get('CurrentBuildNumber') or v.get('CurrentBuild')}, registered owner "
                                  f"{v.get('RegisteredOwner', '-')}, installed {install_utc or '-'}",
                                  [self._rec(rel, k, n) for n in fields],
                                  install_date_utc=install_utc, **{n: render_registry_value(v.get(n)) for n in fields}))
        for base in ("\\Microsoft\\Windows\\CurrentVersion\\Uninstall",
                     "\\Wow6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall"):
            for sub in reg.subkeys(rel, base)[:MAX_PER_KEY]:
                k = f"{base}\\{sub}"
                v = reg.values(rel, k)
                name = v.get("DisplayName")
                if not name or v.get("SystemComponent") == 1:
                    continue
                fields = [n for n in ("DisplayName", "DisplayVersion", "Publisher", "InstallDate", "InstallLocation")
                          if v.get(n) not in (None, "")]
                out.append(self._emit(rel, "win.installed_program",
                                      f"Installed program: {name} {v.get('DisplayVersion', '')} ({v.get('Publisher', '-')}), "
                                      f"install date {v.get('InstallDate', '-')}, key last written {reg.read(rel, k)}",
                                      [self._rec(rel, k, n) for n in fields] + [self._rec(rel, k, None, "key_last_write")],
                                      key=sub, key_last_write_utc=reg.read(rel, k),
                                      **{n: render_registry_value(v.get(n)) for n in fields}))
        k = "\\Microsoft\\Windows NT\\CurrentVersion\\Winlogon"
        v = reg.values(rel, k)
        fields = [n for n in ("DefaultUserName", "DefaultDomainName", "AltDefaultDomainName", "LastUsedUsername")
                  if isinstance(v.get(n), str) and v.get(n)]
        if fields:
            out.append(self._emit(rel, "win.last_logon", "Winlogon: " + ", ".join(f"{n} {v[n]}" for n in fields)
                                  + (f" (last user {v['DefaultUserName']}, domain {v.get('DefaultDomainName', '-')})"
                                     if v.get("DefaultUserName") else ""),
                                  [self._rec(rel, k, n) for n in fields], **{n: v[n] for n in fields}))
        base = "\\Microsoft\\Windows NT\\CurrentVersion\\NetworkCards"
        for sub in reg.subkeys(rel, base)[:MAX_PER_KEY]:
            k = f"{base}\\{sub}"
            v = reg.values(rel, k)
            if v.get("Description"):
                out.append(self._emit(rel, "win.network_card", f"Network card {sub}: {v['Description']}",
                                      [self._rec(rel, k, "Description"), self._rec(rel, k, "ServiceName")],
                                      card=sub, description=v["Description"], service=v.get("ServiceName")))
        base = "\\Microsoft\\Windows NT\\CurrentVersion\\ProfileList"
        for sid in reg.subkeys(rel, base)[:MAX_PER_KEY]:
            k = f"{base}\\{sid}"
            path = reg.values(rel, k).get("ProfileImagePath")
            if path:
                out.append(self._emit(rel, "win.user_profile", f"User profile {sid}: {path}",
                                      [self._rec(rel, k, "ProfileImagePath"), self._rec(rel, k, None, "key_last_write")],
                                      sid=sid, profile_path=path, user=PureWindowsPath(str(path)).name,
                                      key_last_write_utc=reg.read(rel, k)))
        base = "\\Microsoft\\Windows Portable Devices\\Devices"
        for dev in reg.subkeys(rel, base)[:MAX_PER_KEY]:
            k = f"{base}\\{dev}"
            fname = reg.values(rel, k).get("FriendlyName")
            out.append(self._emit(rel, "win.portable_device", f"Portable device '{fname or '-'}': {dev}",
                                  [self._rec(rel, k, "FriendlyName"), self._rec(rel, k, None, "key_last_write")],
                                  device=dev, friendly_name=fname, key_last_write_utc=reg.read(rel, k)))
        base = "\\Microsoft\\Windows NT\\CurrentVersion\\EMDMgmt"
        for dev in reg.subkeys(rel, base)[:MAX_PER_KEY]:
            k = f"{base}\\{dev}"
            label = (re.search(r"\}(.*)_(\d+)$", dev) if "USBSTOR" in dev.upper() else
                     re.match(r"^.{5}_(.*)_(\d+)$", dev))
            out.append(self._emit(rel, "win.portable_device",
                                  f"ReadyBoost record of a volume: {dev}" + (f" (label '{label.group(1)}', serial "
                                                                           f"{label.group(2)})" if label else ""),
                                  [self._rec(rel, k, None, "key_last_write")], device=dev,
                                  volume_label=label.group(1) if label else None,
                                  volume_serial=label.group(2) if label else None, key_last_write_utc=reg.read(rel, k)))
        base = "\\Microsoft\\Windows NT\\CurrentVersion\\NetworkList\\Profiles"
        for guid in reg.subkeys(rel, base)[:MAX_PER_KEY]:
            k = f"{base}\\{guid}"
            v = reg.values(rel, k)
            if v.get("ProfileName"):
                out.append(self._emit(rel, "win.network_profile", f"Network profile '{v['ProfileName']}' ({guid})",
                                      [self._rec(rel, k, "ProfileName"), self._rec(rel, k, "Description")],
                                      profile=v["ProfileName"], description=v.get("Description")))
        return out

    def _sam(self, rel: str) -> list[Finding | None]:
        reg, out = self.bridge.registry, []
        base = "\\SAM\\Domains\\Account\\Users\\Names"
        for user in reg.subkeys(rel, base)[:MAX_PER_KEY]:
            k = f"{base}\\{user}"
            out.append(self._emit(rel, "win.user_profile", f"Local account '{user}' (SAM), key last written {reg.read(rel, k)}",
                                  [self._rec(rel, k, None, "key_last_write")], user=user,
                                  key_last_write_utc=reg.read(rel, k)))
        return out

    def _ntuser(self, rel: str) -> list[Finding | None]:
        reg, out, user = self.bridge.registry, [], self.profile_user(rel)
        ex = "\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer"
        for guid in reg.subkeys(rel, f"{ex}\\UserAssist"):
            k = f"{ex}\\UserAssist\\{guid}\\Count"
            for n, raw in list(reg.values(rel, k).items())[:MAX_PER_KEY]:
                b = _hexbytes(raw)
                program = rot13(n)
                if len(b) >= 68:
                    count, last = struct.unpack("<I", b[4:8])[0], filetime_to_iso(struct.unpack("<Q", b[60:68])[0])
                elif len(b) >= 16:
                    count, last = max(struct.unpack("<I", b[4:8])[0] - 5, 0), filetime_to_iso(struct.unpack("<Q", b[8:16])[0])
                else:
                    continue
                if program.startswith("UEME_CTL") or (not count and not last):
                    continue
                out.append(self._emit(rel, "win.userassist",
                                      f"{user or 'user'} ran {program} {count} time(s), last {last or '-'} (UserAssist)",
                                      [self._rec(rel, k, n)], user=user, program=program, run_count=count,
                                      last_run_utc=last, value_name_rot13=n))
        k = f"{ex}\\RecentDocs"
        for sub in [None, *reg.subkeys(rel, k)]:
            kk = k if sub is None else f"{k}\\{sub}"
            for n, raw in list(reg.values(rel, kk).items())[:MAX_PER_KEY]:
                if n == "MRUListEx":
                    continue
                name = _utf16z(_hexbytes(raw))
                if name:
                    out.append(self._emit(rel, "win.recent_doc", f"{user or 'user'} recently opened '{name}' (RecentDocs"
                                          + (f" {sub}" if sub else "") + ")",
                                          [self._rec(rel, kk, n)], user=user, name=name, extension=sub))
        office = "\\Software\\Microsoft\\Office"
        for ver in reg.subkeys(rel, office):
            for appname in reg.subkeys(rel, f"{office}\\{ver}"):
                for mru in ("File MRU", "Place MRU"):
                    kk = f"{office}\\{ver}\\{appname}\\{mru}"
                    for n, v in reg.values(rel, kk).items():
                        if isinstance(v, str) and n.lower().startswith("item"):
                            path = v.split("*", 1)[-1]
                            out.append(self._emit(rel, "win.recent_doc", f"{user or 'user'} opened '{path}' in {appname} "
                                                  f"(Office {ver} {mru})", [self._rec(rel, kk, n)], user=user,
                                                  name=path, application=appname))
        kk = "\\Software\\Microsoft\\Internet Explorer\\TypedURLs"
        for n, v in reg.values(rel, kk).items():
            if isinstance(v, str) and v:
                out.append(self._emit(rel, "win.typed_url", f"{user or 'user'} typed URL {v}", [self._rec(rel, kk, n)],
                                      user=user, url=v))
        kk = f"{ex}\\WordWheelQuery"
        for n, raw in reg.values(rel, kk).items():
            if n == "MRUListEx":
                continue
            term = _utf16z(_hexbytes(raw))
            if term:
                out.append(self._emit(rel, "win.search_term", f"{user or 'user'} searched Explorer for '{term}'",
                                      [self._rec(rel, kk, n)], user=user, term=term))
        kk = f"{ex}\\RunMRU"
        for n, v in reg.values(rel, kk).items():
            if n != "MRUList" and isinstance(v, str):
                out.append(self._emit(rel, "win.run_mru", f"{user or 'user'} ran '{v.rstrip(chr(92) + '1')}' from Run",
                                      [self._rec(rel, kk, n)], user=user, command=v))
        kk = f"{ex}\\MountPoints2"
        for sub in reg.subkeys(rel, kk)[:MAX_PER_KEY]:
            out.append(self._emit(rel, "win.mountpoint", f"{user or 'user'} accessed volume / share {sub.replace('#', chr(92))}",
                                  [self._rec(rel, f"{kk}\\{sub}", None, "key_last_write")], user=user, mountpoint=sub,
                                  key_last_write_utc=reg.read(rel, f"{kk}\\{sub}")))
        return out

    # ------------------------------------------------------------------ binary artefacts
    def _read(self, rel: str, limit: int = 64 << 20) -> bytes:
        with open(self.sandbox.resolve(rel), "rb") as fh:
            return fh.read(limit)

    def _prefetch(self, rel: str) -> list[Finding]:
        data = self._read(rel)
        pf = parse_prefetch(data)
        if not pf:
            return []
        recs = [bytes_record(self.sandbox, rel, *pf["name_bytes"], "executable", pf["executable"]),
                bytes_record(self.sandbox, rel, *pf["run_count_bytes"], "run_count", pf["run_count"])]
        if pf["last_run"]:
            recs.append(bytes_record(self.sandbox, rel, *pf["last_run_bytes"], "last_run", pf["last_run"]))
        f = self._emit(rel, "win.prefetch", f"{pf['executable']} executed {pf['run_count']} time(s), last {pf['last_run'] or '-'} "
                       f"(Prefetch v{pf['version']})", recs, executable=pf["executable"], run_count=pf["run_count"],
                       last_run_utc=pf["last_run"], version=pf["version"])
        return [f] if f else []

    def _lnk(self, rel: str) -> list[Finding]:
        import LnkParse3

        data = self._read(rel, 4 << 20)
        if data[:4] != b"L\x00\x00\x00":
            return []
        import io
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # LnkParse3 warns on every unknown shell-item variant
            info = LnkParse3.lnk_file(io.BytesIO(data)).get_json()
        li = info.get("link_info") or {}
        loc = li.get("location_info") or {}
        target = li.get("local_base_path") or ""
        suffix = li.get("common_path_suffix") or ""
        net = loc.get("net_name") or ""
        full = target or (f"{net}\\{suffix}" if net else "") or (info.get("data") or {}).get("relative_path", "")
        recs = [bytes_record(self.sandbox, rel, 0x1C, data[0x1C:0x34], "target_times_raw")]
        for text, field in ((target, "local_base_path"), (suffix, "common_path_suffix"), (net, "net_name"),
                            (loc.get("volume_label") or "", "volume_label"),
                            ((info.get("data") or {}).get("relative_path") or "", "relative_path")):
            hit = find_string(data, text)
            if hit:
                recs.append(bytes_record(self.sandbox, rel, hit[0], hit[1], field, text))
        if len(recs) == 1:
            return []
        times = {k: filetime_to_iso(struct.unpack("<Q", data[o:o + 8])[0])
                 for k, o in (("target_created", 0x1C), ("target_accessed", 0x24), ("target_modified", 0x2C))}
        serial = loc.get("drive_serial_number")
        f = self._emit(rel, "win.lnk",
                       f"Shortcut {PurePosixPath(self.logical(rel)).name} -> {full or '?'} "
                       f"({loc.get('drive_type') or li.get('location') or '-'}"
                       + (f", volume '{loc.get('volume_label')}' serial {serial}" if serial else "")
                       + f"), target modified {times['target_modified'] or '-'}",
                       recs, target=full, drive_type=loc.get("drive_type"), drive_serial=serial,
                       volume_label=loc.get("volume_label"), net_name=net or None, user=self.profile_user(rel),
                       target_size=info.get("header", {}).get("file_size"), **times)
        return [f] if f else []

    def _recycle_i(self, rel: str) -> list[Finding]:
        data = self._read(rel, 1 << 16)
        if len(data) < 28:
            return []
        version = struct.unpack("<Q", data[:8])[0]
        size = struct.unpack("<Q", data[8:16])[0]
        deleted = filetime_to_iso(struct.unpack("<Q", data[16:24])[0])
        if version == 2:
            n = struct.unpack("<I", data[24:28])[0]
            off, raw = 28, data[28:28 + 2 * n]
        else:
            off, raw = 24, data[24:24 + 520]
        path = _utf16z(raw)
        if not path:
            return []
        recs = [bytes_record(self.sandbox, rel, 8, data[8:24], "size_and_deleted_time"),
                bytes_record(self.sandbox, rel, off, path.encode("utf-16-le"), "original_path", path)]
        f = self._emit(rel, "win.recycle_bin", f"Deleted to Recycle Bin: {path} ({size} bytes) at {deleted or '-'}",
                       recs, original_path=path, size=size, deleted_utc=deleted)
        return [f] if f else []

    def _info2(self, rel: str) -> list[Finding]:
        """XP Recycle Bin index: 20-byte header (record size at 12), records of ANSI path (260), index, drive,
        deletion FILETIME (+268), size (+276) and Unicode path (+280, 520 bytes)."""
        data = self._read(rel, 16 << 20)
        if len(data) < 20:
            return []
        rec_size = struct.unpack("<I", data[12:16])[0]
        if rec_size not in (280, 800):
            return []
        out = []
        for off in range(20, len(data) - rec_size + 1, rec_size):
            r = data[off:off + rec_size]
            ansi = r[:260].split(b"\x00", 1)[0].decode("cp1252", "replace")
            idx, drive = struct.unpack("<II", r[260:268])
            deleted = filetime_to_iso(struct.unpack("<Q", r[268:276])[0])
            size = struct.unpack("<I", r[276:280])[0]
            path = _utf16z(r[280:800]) if rec_size == 800 else ""
            path = path or ansi
            if not path:
                continue
            recs = [bytes_record(self.sandbox, rel, off, r[:len(ansi.encode("cp1252", "replace"))] or r[:1], "ansi_path", ansi),
                    bytes_record(self.sandbox, rel, off + 260, r[260:280], "index_drive_time_size")]
            out.append(self._emit(rel, "win.recycle_bin",
                                  f"Deleted to Recycle Bin: {path} ({size} bytes) at {deleted or '-'} (INFO2 record {idx}, "
                                  f"stored as D{chr(ord('a') + drive) if drive < 26 else '?'}{idx})",
                                  recs, original_path=path, size=size, deleted_utc=deleted, record_index=idx))
        return [f for f in out if f]

    def _index_dat(self, rel: str) -> list[Finding]:
        data = self._read(rel)
        if not data.startswith(b"Client UrlCache MMF"):
            return []
        out, seen = [], set()
        for m in IE_URL_RECORD.finditer(data):
            rec = m.start()
            if rec % 0x80 or rec + 0x68 > len(data):
                continue                       # records are 128-byte aligned
            loc_off = struct.unpack("<I", data[rec + 0x34:rec + 0x38])[0]
            if not 0x40 <= loc_off < 0x400:
                continue
            start = rec + loc_off
            end = data.find(b"\x00", start, start + 2048)
            if end <= start:
                continue
            loc = data[start:end].decode("cp1252", "replace")
            if not re.search(r"(https?|ftp|file)://|^:\d{10}: |^Visited: ", loc) or loc in seen:
                continue
            seen.add(loc)
            accessed = filetime_to_iso(struct.unpack("<Q", data[rec + 0x10:rec + 0x18])[0])
            url = re.sub(r"^(Visited: |:\d{10}: )", "", loc)
            user, _, url = url.rpartition("@") if re.match(r"^[^/:@]+@", url) else ("", "", url)
            out.append(self._emit(rel, "ie.url", f"Internet Explorer record: {url} (user {user or '-'}), accessed {accessed or '-'}",
                                  [bytes_record(self.sandbox, rel, start, data[start:end], "location", loc),
                                   bytes_record(self.sandbox, rel, rec + 0x08, data[rec + 0x08:rec + 0x18], "times_raw")],
                                  url=url, user=user or self.profile_user(rel), accessed_utc=accessed, raw_location=loc))
            if len(out) >= 2000:
                break
        return [f for f in out if f]

    # ------------------------------------------------------------------ event logs
    def _evtx(self, rel: str) -> list[Finding]:
        low = PurePosixPath(self.logical(rel).lower()).name
        wanted: dict[int, str] = {}
        provider = None
        for chan, ids in EVENT_IDS.items():
            if chan in low:
                wanted.update(ids)
                provider = EVENT_PROVIDER.get(chan, provider)
        if not wanted:
            return []
        out = []
        for rid, xml in self.bridge.evtx.records(rel):
            m = re.search(r"<EventID[^>]*>(\d+)</EventID>", xml)
            if not m or int(m.group(1)) not in wanted:
                continue
            eid = int(m.group(1))
            if provider and f'Provider Name="{provider}"' not in xml:
                continue
            if eid == 4624:
                lt = re.search(r'<Data Name="LogonType">(\d+)</Data>', xml)
                if not lt or lt.group(1) not in _LOGON_TYPES:
                    continue
            rec = self.bridge.evtx.record(rel, rid, xml, f"event.{eid}")
            user = re.search(r'<Data Name="TargetUserName">([^<]*)</Data>', xml)
            when = re.search(r'SystemTime="([^"]*)"', xml)
            out.append(self._emit(rel, "win.event", f"{wanted[eid]} (event {eid}) at {when.group(1) if when else '-'}"
                                  + (f", user {user.group(1)}" if user else "") + f": {rec.raw_text_snippet[:200]}",
                                  [rec], event_id=eid, meaning=wanted[eid], record_id=rid,
                                  time_utc=when.group(1) if when else None, user=user.group(1) if user else None,
                                  channel=low.removesuffix(".evtx")))
            if len(out) >= MAX_EVENTS_PER_LOG:
                break
        return [f for f in out if f]


def run_on_evidence(case, evidence_id: str, tracer=None, context=None, memory=None, run_id=None,
                    output_dir: Path | None = None, limit: int = 0):
    from src.bridge.forensic_bridge import ForensicBridge

    ev_dir = case.dir / "evidence" / evidence_id
    manifest = json.loads((ev_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    bridge = ForensicBridge(ev_dir / "fs")
    agent = None
    try:
        agent = WindowsArtifactAgent(bridge, manifest=manifest, tracer=tracer, context=context, memory=memory,
                                     evidence_id=evidence_id, run_id=run_id)
        files = agent.discover()
        if limit:
            files = files[:limit]
        return agent.run(files, output_dir=output_dir) if files else None
    finally:
        if agent is not None:
            agent.close()
        bridge.close()

