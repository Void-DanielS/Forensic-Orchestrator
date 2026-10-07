"""Case registry: every investigation, its evidence items and its context documents.

A **case** is one investigation. It groups every evidence item seized for that scenario
(computers, removable media, phones, cloud exports) and the documents describing it:

* ``brief``       what an examiner is told at the start (warrant, police reports, scenario
                  description, image-creation notes). Agents read these and test them
                  against the evidence.
* ``answer_key``  solutions, teacher guides, instructor packets, creation logs. **Never shown
                  to agents**; used only to score their output (and for supervised training).
* ``acquisition`` imaging logs (FTK Imager / EWF headers): chain of custody, not narrative.

Layout on disk::

    data/cases/
    ├── catalog.sqlite                  every case / evidence item / extracted file (``main.py cases catalog``)
    └── <case_id>/
        ├── case.json                   this registry entry + build status
        ├── context/brief/              + extracted .txt next to each document
        ├── context/answer_key/
        ├── evidence/<evidence_id>/     fs/ (agent sandbox), MANIFEST.json, INSPECTION.json, FILELIST.jsonl.gz
        └── workspace/                  agent memory, run traces, reports
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config.settings import DATA_DIR
from src.utils.disk_image import DiskEvidence

CASES_DIR = DATA_DIR / "cases"
M57 = "scenarios/2009-m57-patents"
LW = "scenarios/2018-lonewolf"
NARCOS = "scenarios/2019-narcos"
OWL = "scenarios/2019-owl"
NIST = "https://cfreds-archive.nist.gov/data_leakage_case"
MAG = "scenarios/magnet"


@dataclass(frozen=True)
class Doc:
    url: str                      # absolute or bucket-relative; for a ZIP, the archive
    role: str                     # brief | answer_key | acquisition
    members: str = ""             # regex of ZIP members to keep (when url is a ZIP)
    rename: str = ""              # local file name (e.g. for an HTML page)


@dataclass(frozen=True)
class Case:
    case_id: str
    title: str
    year: int
    kind: str                     # scenario | ctf | reference_device | challenge
    summary: str                  # neutral one-paragraph description (no solutions)
    origin: str
    evidence: tuple = ()          # DiskEvidence objects or mobile source ids (str, see dataset_scraper.SOURCES)
    docs: tuple[Doc, ...] = ()
    questions: tuple[str, ...] = ()   # investigative questions given with the case, if any
    tags: tuple[str, ...] = field(default_factory=tuple)
    publisher_notes: str = ""         # the publisher's own description (catalogue page), saved as a brief doc
    source_url: str = ""              # landing page of the case at its publisher
    answer_notes: str = ""            # publisher text that reveals the solution / plot: saved to answer_key only

    @property
    def dir(self) -> Path:
        return CASES_DIR / self.case_id

    def evidence_ids(self) -> list[str]:
        return [e if isinstance(e, str) else e.evidence_id for e in self.evidence]


def evidence_meta(e) -> dict:
    """Uniform view of a mobile source id, a DiskEvidence or a RemoteEvidence."""
    from src.utils import dataset_scraper as ds

    if isinstance(e, str):
        s = ds.SOURCES[e]
        return {"evidence_id": e, "kind": "mobile", "title": s.title, "platform": s.platform, "group": s.group,
                "transfer_gb": s.transfer_gb, "device": "", "note": s.note}
    kind = "bundle" if type(e).__name__ == "RemoteEvidence" else "disk_image"
    return {"evidence_id": e.evidence_id, "kind": kind, "title": e.title, "platform": e.platform, "group": e.group,
            "transfer_gb": e.transfer_gb, "device": e.device, "note": e.note}


def _m57(eid: str, img: str, desc: str, gb: float, group: str = "light") -> DiskEvidence:
    return DiskEvidence(eid, f"M57 {desc}",
                        "removable" if "usb" in img else "windows", urls=(f"{M57}/drives-redacted/{img}.E01",),
                        device=desc, group=group, transfer_gb=gb)


_HICK = "Joshua Hickman reference image (Digital Corpora)"
_MAG = "Magnet Forensics CTF (Digital Corpora)"

_CORE_CASES: dict[str, Case] = {c.case_id: c for c in (
    # ================================================================== computer-centred scenarios
    Case("m57_patents", "M57-Patents: exfiltration, eavesdropping and contraband at a start-up", 2009, "scenario",
         "M57.biz, a small patent-research start-up, ran for one month (Nov-Dec 2009). Four employees (Pat, the "
         "CEO; Terry, IT; Jo and Charlie, patent researchers) used Windows workstations and USB drives. Police "
         "seized the computers and USB drives after a lead about stolen proprietary information and illegal "
         "content. Images are the final-day (2009-12-11) snapshots.",
         "Naval Postgraduate School / Digital Corpora",
         evidence=(
             _m57("m57_charlie", "charlie-2009-12-11", "Charlie's workstation (Windows XP), 2009-12-11", 3.9),
             _m57("m57_jo", "jo-2009-12-11-002", "Jo's workstation (Windows XP), 2009-12-11", 5.9),
             _m57("m57_pat", "pat-2009-12-11", "Pat's (CEO) workstation (Windows XP), 2009-12-11", 6.1),
             _m57("m57_terry", "terry-2009-12-11-002", "Terry's (IT) workstation (Windows), 2009-12-11", 10.6),
             _m57("m57_charlie_usb", "charlie-work-usb-2009-12-11", "Charlie's work USB drive", 0.01),
             _m57("m57_jo_favorites_usb", "jo-favorites-usb-2009-12-11", "Jo's 'favorites' USB drive", 0.23),
             _m57("m57_jo_work_usb", "jo-work-usb-2009-12-11", "Jo's work USB drive", 0.12),
             _m57("m57_terry_usb", "terry-work-usb-2009-12-11", "Terry's work USB drive", 0.03),
         ),
         docs=(Doc(f"{M57}/docs/m57-affidavit-warrant-final.pdf", "brief"),
               *(Doc(f"{M57}/docs/detectivereport{i}.pdf", "brief") for i in (1, 2, 3, 4)),
               Doc(f"{M57}/docs/m57-instructor-packet.pdf", "answer_key"),
               Doc(f"{M57}/docs/M57-Patents-Exfiltration.pdf", "answer_key"),
               Doc(f"{M57}/docs/M57-Patents-Eavesdropping.pdf", "answer_key"),
               Doc(f"{M57}/docs/M57-Patents-Illegal.pdf", "answer_key")),
         tags=("windows", "usb", "multi-device", "exfiltration")),

    Case("lone_wolf", "Lone Wolf scenario", 2018, "scenario",
         "A suspect's Windows laptop was seized and imaged (with memory and pagefile captures). The scenario "
         "README describes the situation given to the examiner.",
         "Digital Corpora (2018 Lone Wolf scenario)",
         evidence=(DiskEvidence("lonewolf_laptop", "Lone Wolf suspect laptop (Windows 10)", "windows",
                                urls=tuple(f"{LW}/LoneWolf_Image_Files/LoneWolf.E0{i}" for i in range(1, 10)),
                                acquisition_docs=(f"{LW}/LoneWolf_Image_Files/FTK Imager Log.txt",),
                                device="Suspect laptop, Windows 10", transfer_gb=3.0),),
         docs=(Doc(f"{LW}/0  README.pdf", "brief"),
               Doc(f"{LW}/Lone Wolf Teacher Guide.zip", "answer_key", members=r"\.(pdf|docx)$"),
               Doc(f"{LW}/Lone Wolf Forensic Reports.zip", "answer_key", members=r"\.(pdf|docx|csv|txt|html)$")),
         tags=("windows", "single-device", "terrorism")),

    Case("narcos", "Narcos scenario", 2019, "scenario",
         "Three Windows computers imaged for the Narcos scenario (memory images are labelled John, Jane and "
         "Steve), plus one computer imaged again after CCleaner was run.",
         "Digital Corpora (2019 Narcos scenario)",
         evidence=(
             DiskEvidence("narcos_1", "Narcos computer 1", "windows", zip_url=f"{NARCOS}/Narcos-1.zip",
                          member_regex=r"Image/Narcos-1\.\d{3}$", fmt="raw", acquisition_docs=(r"\.001\.txt$",),
                          group="heavy", transfer_gb=9.7),
             DiskEvidence("narcos_2", "Narcos computer 2", "windows", zip_url=f"{NARCOS}/Narcos-2.zip",
                          member_regex=r"Image/Narcos-2\.\d{3}$", fmt="raw", acquisition_docs=(r"\.001\.txt$",),
                          group="heavy", transfer_gb=9.9),
             DiskEvidence("narcos_3", "Narcos computer 3", "windows", zip_url=f"{NARCOS}/Narcos-3.zip",
                          member_regex=r"Image/Narcos-3\.\d{3}$", fmt="raw", acquisition_docs=(r"\.001\.txt$",),
                          group="heavy", transfer_gb=7.7),
             DiskEvidence("narcos_ccleaner", "Narcos computer after CCleaner", "windows",
                          urls=(f"{NARCOS}/Narcos-CCleaner.E01",), acquisition_docs=(f"{NARCOS}/Narcos-CCleaner.001.txt",),
                          transfer_gb=3.0, note="anti-forensics: wiped with CCleaner before imaging"),
         ),
         docs=(Doc(f"{NARCOS}/Narcos_teacher_guide.zip", "answer_key", members=r"\.(pdf|xlsx)$"),),
         tags=("windows", "multi-device", "drugs", "anti-forensics"),
         publisher_notes="Due to intelligence provided by the Australian government, two passengers were "
                         "intercepted by Customs upon arriving at Wellington, New Zealand from Brisbane.\n\n"
                         "You are a Customs forensics investigator. Customs officers have delivered a forensic image "
                         "and memory dump of the suspect's desktop computer to you. Your task is to determine the "
                         "relationship between John Fredrickson and the suspect, their future intentions, and any "
                         "other supporting evidence that pertains to the case.",
         source_url="https://digitalcorpora.org/corpora/scenarios/"),

    Case("nist_data_leakage", "NIST CFReDS Data Leakage: confidential data taken by an employee", 2015, "scenario",
         "An employee (Mr. Informant) of a company is suspected of leaking confidential information to a "
         "competitor ('spy'). His work PC, two USB storage devices and a CD-R were seized.",
         "NIST CFReDS",
         evidence=(
             DiskEvidence("nist_pc", "Informant's PC (Windows 7)", "windows",
                          urls=tuple(f"{NIST}/images/pc/cfreds_2015_data_leakage_pc.E0{i}" for i in range(1, 5)),
                          device="Suspect work PC", transfer_gb=3.0),
             DiskEvidence("nist_rm1", "Removable media #1 (USB)", "removable",
                          urls=(f"{NIST}/images/rm%231/cfreds_2015_data_leakage_rm%231.E01",), transfer_gb=0.5),
             DiskEvidence("nist_rm2", "Removable media #2 (USB)", "removable",
                          urls=(f"{NIST}/images/rm%232/cfreds_2015_data_leakage_rm%232.E01",), transfer_gb=0.5),
             DiskEvidence("nist_rm3", "Removable media #3 (CD-R)", "removable",
                          urls=(f"{NIST}/images/rm%233/cfreds_2015_data_leakage_rm%233_type3.E01",), transfer_gb=0.3,
                          group="future", note="pure UDF 2.x file system (no ISO-9660 bridge): The Sleuth Kit cannot "
                                               "read it; needs a UDF reader"),
         ),
         docs=(Doc(f"{NIST}/data-leakage-case.html", "brief", rename="data-leakage-case.html"),
               Doc(f"{NIST}/leakage-answers.pdf", "answer_key")),
         tags=("windows", "usb", "cd", "multi-device", "data-leakage")),

    Case("owl", "Owl scenario (Project Owl)", 2019, "scenario",
         "A desktop hard disk (HD1) and an LG Nexus 5 phone imaged for the same scenario.",
         "Digital Corpora (2019 Owl scenario)",
         evidence=(DiskEvidence("owl_hd1", "Owl desktop hard disk HD1", "windows", urls=(f"{OWL}/HD1/HD1.E01",),
                                acquisition_docs=(f"{OWL}/HD1/HD1.log",), transfer_gb=4.0),
                   "owl_nexus5"),
         docs=(Doc(f"{OWL}/Documentation from Creation of Owl Scenario.zip", "answer_key", members=r"\.docx$"),),
         tags=("windows", "android", "multi-device")),

    Case("dfrws2021", "DFRWS 2021: card skimming, 3D printer and NAS", 2021, "challenge",
         "Multiple devices seized in linked criminal activities: a card-skimmer microSD, a Raspberry Pi "
         "(3D printer controller) microSD and a Samsung Galaxy S10. Goal: multisource correlation.",
         "DFRWS 2021 Forensic Challenge",
         evidence=(DiskEvidence("dfrws_skimmer", "Card skimmer microSD", "removable",
                                zip_url="dfrws/challenge-2021/1_Skimmer_mSD.zip", member_regex=r"\.e01$",
                                acquisition_docs=(r"\.txt$",), transfer_gb=0.04),
                   DiskEvidence("dfrws_rpi", "Raspberry Pi (3D printer) microSD", "linux",
                                zip_url="dfrws/challenge-2021/2_Raspberry_Pi_mSD.zip", member_regex=r"\.e01$",
                                acquisition_docs=(r"\.txt$",), group="heavy", transfer_gb=1.8),
                   "dfrws2021_galaxy_s10"),
         docs=(Doc("https://raw.githubusercontent.com/dfrws/dfrws2021-challenge/main/README.md", "brief"),
               Doc("https://raw.githubusercontent.com/dfrws/dfrws2021-challenge/main/challenge-details.md", "brief")),
         tags=("iot", "linux", "android", "multi-device")),

    # ================================================================== Magnet CTF (one storyline per year)
    Case("magnet_ctf_2019", "Magnet CTF 2019", 2019, "ctf",
         "Magnet Forensics 2019 CTF: a Windows desktop and an Android phone from the same scenario.",
         _MAG,
         evidence=(DiskEvidence("magnet2019_windows", "Magnet CTF 2019 Windows desktop", "windows",
                                zip_url=f"{MAG}/2019 CTF - Windows-Desktop.zip", member_regex=r"\.E01$",
                                acquisition_docs=(r"\.E01\.txt$",), transfer_gb=3.0),
                   "magnet2019_android"),
         docs=(Doc(f"{MAG}/README.md", "brief"),), tags=("windows", "android", "multi-device", "ctf")),
    Case("magnet_ctf_2020", "Magnet CTF 2020", 2020, "ctf",
         "Magnet Forensics 2020 CTF: Windows computer, Android phone, iPhone and Google Takeout of the same people.",
         _MAG,
         evidence=(DiskEvidence("magnet2020_windows", "Magnet CTF 2020 Windows computer", "windows",
                                zip_url=f"{MAG}/2020 CTF - Windows.zip", member_regex=r"DFA_Windows\.E\d\d$",
                                acquisition_docs=(r"\.E01\.txt$",), group="heavy", transfer_gb=29.3),
                   "magnet2020_android", "magnet2020_ios", "magnet2020_takeout"),
         docs=(Doc(f"{MAG}/README.md", "brief"),),
         tags=("windows", "android", "ios", "cloud", "multi-device", "ctf")),
    Case("magnet_ctf_2021", "Magnet CTF 2021", 2021, "ctf",
         "Magnet Forensics 2021 CTF: iPhone 8 ('Eli') full file system (Chromebook and macOS images registered).",
         _MAG, evidence=("magnet2021_ios",), docs=(Doc(f"{MAG}/README.md", "brief"),),
         tags=("ios", "ctf")),
    Case("magnet_ctf_2022", "Magnet CTF 2022", 2022, "ctf",
         "Magnet Forensics 2022 CTF: HP Windows laptop, Android phone and iPhone 8 ('Jess').",
         _MAG,
         evidence=(DiskEvidence("magnet2022_windows", "Magnet CTF 2022 HP laptop (Windows)", "windows",
                                zip_url=f"{MAG}/2022 CTF - Windows.zip", member_regex=r"Laptop1Final\.E\d\d$",
                                group="heavy", transfer_gb=37.7),
                   "magnet2022_android", "magnet2022_ios"),
         docs=(Doc(f"{MAG}/README.md", "brief"),), tags=("windows", "android", "ios", "multi-device", "ctf")),

    # ================================================================== reference devices (one device = one case)
    *(Case(f"hickman_{sid}", f"Reference device: {title}", year, "reference_device",
           f"{title} populated by the examiner with a documented set of apps and user activity. The image-creation "
           "document lists what was done and when; it is the investigation context.", _HICK, evidence=(sid,),
           tags=(platform, "single-device", "reference"),
           publisher_notes=f"An image of a {title} was created by Joshua Hickman: popular applications were "
                           "populated with user data using each app's own capabilities (messaging, browsing, maps, "
                           "photos, social media, games), over a documented period. Reconstruct the user's "
                           "activity: accounts, communications, locations, media and app usage, with times.",
           source_url="https://thebinaryhick.blog/public_images/")
      for sid, title, year, platform in (
          ("android_9", "Google Pixel 3, Android 9", 2019, "android"),
          ("android_10", "Google Pixel 3, Android 10", 2020, "android"),
          ("android_13", "Android 13 phone", 2022, "android"),
          ("android_14", "Android 14 phone", 2024, "android"),
          ("ios13_3_1_ffs", "iPhone, iOS 13.3.1", 2020, "ios"),
          ("ios13_4_1_ffs", "iPhone, iOS 13.4.1", 2020, "ios"),
          ("ios16", "iPhone, iOS 16", 2023, "ios"),
          ("ios17", "iPhone, iOS 17", 2024, "ios"))),
)}


def _all_cases() -> dict[str, Case]:
    from src.cases.registry_ext import EXT_CASES

    merged = dict(_CORE_CASES)
    for c in EXT_CASES:
        if c.case_id in merged:
            raise ValueError(f"duplicate case id {c.case_id}")
        merged[c.case_id] = c
    seen: dict[str, str] = {}
    for c in merged.values():
        for eid in c.evidence_ids():
            if eid in seen:
                raise ValueError(f"evidence id {eid} used by {seen[eid]} and {c.case_id}")
            seen[eid] = c.case_id
    return merged


CASES: dict[str, Case] = _all_cases()


def case_of_evidence(evidence_id: str) -> Case | None:
    for c in CASES.values():
        if evidence_id in c.evidence_ids():
            return c
    return None


def evidence_dir(evidence_id: str) -> Path:
    c = case_of_evidence(evidence_id)
    if c is None:
        raise KeyError(f"evidence '{evidence_id}' belongs to no case")
    return c.dir / "evidence" / evidence_id


def disk_evidence(evidence_id: str) -> DiskEvidence | None:
    for c in CASES.values():
        for e in c.evidence:
            if isinstance(e, DiskEvidence) and e.evidence_id == evidence_id:
                return e
    return None
