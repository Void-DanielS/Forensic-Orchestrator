"""Case registry, part 2: the extended public case collection (Sprint 1.6).

Every case here has (a) at least one evidence item that is a forensic image or acquisition
(disk / phone / memory / network / IoT dump) and (b) context an examiner would be given: the
publisher's scenario text, image-creation / data-population notes or challenge questions.
Solutions, write-ups, challenge results and instructor material are registered as
``answer_key`` and never shown to agents.

Evidence is described with :class:`~src.utils.bundle.RemoteEvidence` (any host, any container).
``publisher_notes`` keep only the scenario / task text of a publisher page, never its results;
text that reveals the plot goes to ``answer_notes``.

Archive passwords below are the ones the publishers print next to the download links (CFReDS
catalogue / case pages). They open the distribution archives only; they are never written to a
manifest or a report.
"""
from __future__ import annotations

from src.cases.registry import Case, Doc
from src.utils.bundle import RemoteEvidence as RE

DC_ORIGIN = "Digital Corpora"
DFRWS = "DFRWS Forensic Challenge"
DFRWS_ANSWERS = r"(^|/)(README\.md$|results?/|submissions?/|challenge-submissions/|solutions?/|winners?)"
HICKMAN = "Joshua Hickman reference image (Digital Corpora)"
NIST = "NIST CFReDS"
CFTT_MOBILE = "https://s3.amazonaws.com/cftt.cfreds.nist.gov/cfreds/mobile/"
AHADI = "Ali Hadi (ashemery.com DFIR challenges, archive.org)"
VTO = "VTO Labs Drone Forensics Program (DHS S&T), via NIST CFReDS"
DRONES = "https://drive.google.com/drive/folders/1-UrxFGpCo54bVujwFmmqNbsZEV28dSNz"


def _dfrws(year, title, tags, notes, evidence, questions=(), case_id=None, docs=()):
    return Case(case_id or f"dfrws{year}", f"DFRWS {year} Challenge: {title}", year, "challenge",
                notes.split("\n\n")[0][:600], DFRWS, evidence=evidence, docs=docs, questions=questions,
                tags=("dfrws", *tags), publisher_notes=notes,
                source_url=f"https://github.com/dfrws/dfrws{year}-challenge")


# ============================================================================ NIST CFTT mobile (chip-off / JTAG)
_CFTT = (
    # case id, device, platform note, [(path, kind)], population doc
    ("htc_desire_626", "HTC Desire 626s", ["CHIPOFF/HTC+Desire+626/N115018+CHIP+OFF.001"],
     "CHIPOFF/HTC+Desire+626/Mobile+Device+Data+Population_HTC+Desire+626s.pdf"),
    ("moto_e", "Motorola Moto E", ["CHIPOFF/Moto-E/N115015+CHIP+OFF.001"],
     "CHIPOFF/Moto-E/Mobile+Device+Data+Population_MOTO-E.pdf"),
    ("lg_k7", "LG K7", ["CHIPOFF/LG+K7/Chipoff.001"], "CHIPOFF/LG+K7/Mobile+Device+Data+Population_LGK7.docx"),
    ("samsung_s2", "Samsung Galaxy S2", ["CHIPOFF/Samsung+S2/N115020.001"],
     "CHIPOFF/Samsung+S2/Mobile+Device+Data+Population_SamsungGalaxyS2_1.docx"),
    ("zte_z970", "ZTE Z970", ["CHIPOFF/ZTE+Z970/Chipoff.001"],
     "CHIPOFF/ZTE+Z970/Android+Z970_Mobile+Device+Data+Population.docx"),
    ("samsung_s4", "Samsung Galaxy S4 (chip-off and JTAG)",
     ["CHIPOFF/Samsung+S4/N116133.001", "JTAG/Samsung+S4/samsungS4_M919.bin"],
     "CHIPOFF/Samsung+S4/Mobile+Device+Data+Population_samsungGalaxyS4_N116133.docx"),
    ("htc_desire_s", "HTC Desire S (chip-off and JTAG)",
     ["CHIPOFF/HTC+Desire+S/HTC_Desire_S.img", "JTAG/HTC+Desire+S/HTC_Desire_S_JTAG.bin"],
     "JTAG/HTC+Desire+S/Mobile+Device+Data+Population_HTC_Desire_S.pdf"),
    ("htc_one_xl", "HTC One XL (chip-off and JTAG)",
     ["CHIPOFF/HTC%20One%20XL/HTC_One_XL.img", "JTAG/HTC+One+XL/HTC_One_XL_JTAG.bin"],
     "JTAG/HTC+One+XL/Mobile+Device+Data+Population_HTCOne_XL.pdf"),
    ("lg_optimus_hub", "LG E510 Optimus Hub (JTAG)", ["JTAG/LG+Optimus/LG_E510_OPTIMUS_HUB_JTAG.bin"],
     "JTAG/LG+Optimus/Mobile+Device+Data+Population_LG_Optimus_HUB.pdf"),
    ("htc_one_mini", "HTC One Mini (chip-off)", ["CHIPOFF/HTC+One+Mini/HTC_One_Mini.img", "CHIPOFF/HTCOne/HTCOneMini.bin"],
     "CHIPOFF/HTCOne/Mobile+Device+Data+Population_HTCOneMini_N116148.docx"),
)

_CFTT_CASES = tuple(
    Case(f"cftt_{cid}", f"NIST CFTT reference phone: {dev}", 2018, "reference_device",
         f"A {dev} populated by NIST CFTT with a documented set of contacts, calls, messages, media and app data, then "
         "acquired by chip-off and/or JTAG. The data-population document lists what was put on the device.",
         NIST, evidence=(RE(f"cftt_{cid}_dump", f"{dev}: flash memory dump", "android",
                            tuple(CFTT_MOBILE + u for u in urls), device=dev, transfer_gb=8.0, group="heavy",
                            max_artifact_bytes=1_200_000_000),),
         docs=(Doc(CFTT_MOBILE + doc, "brief"),), tags=("android", "chip-off", "jtag", "reference", "single-device"),
         publisher_notes=f"The following binary images were created by performing either a JTAG or Chip-off data "
                         f"extraction technique on a {dev}. The data-population document (context/brief) documents "
                         "the data contents populated onto the device. The binary image can be imported into "
                         "digital forensic tools capable of parsing JTAG and/or chip-off binary files.",
         source_url="https://cfreds-archive.nist.gov/mobile/index.html")
    for cid, dev, urls, doc in _CFTT)


# ============================================================================ VTO drones (one model = one case)
_DRONE_MODELS = (
    # case id, model title, drone folder in the shared tree, report file name, platform note
    ("dji_phantom_3", "DJI Phantom 3", "DJI_Phantom_3/df002_DJI_Phantom_3/", "DJI_Phantom_3.pdf"),
    ("dji_phantom_4", "DJI Phantom 4 Professional", "DJI_Phantom_4/df005_DJI_Phantom_4/", "DJI_Phantom_4.pdf"),
    ("dji_mavic_pro", "DJI Mavic Pro", "DJI_Mavic_Pro/df020_DJI_Mavic_Pro/", "DJI_Mavic_Pro.pdf"),
    ("dji_mavic_air", "DJI Mavic Air", "DJI_Mavic_Air/df048_DJI_Mavic_Air/", "DJI_Mavic_Air.pdf"),
    ("dji_spark", "DJI Spark", "DJI_Spark/df008_DJI_Spark/", "DJI_Spark.pdf"),
    ("dji_inspire_2", "DJI Inspire 2", "DJI_Inspire_2/df027_DJI_Inspire_2/", "DJI_Inspire_2.pdf"),
    ("dji_matrice_210", "DJI Matrice 210", "DJI_Matrice_210/df060_DJI_Matrice_210/", "DJI_Matrice_210.pdf"),
    ("dji_mavic_2_zoom", "DJI Mavic 2 Zoom", "DJI_Mavic_2/df068_DJI_Mavic_2_Zoom/", "DJI_Mavic_2_Zoom.pdf"),
    ("parrot_anafi", "Parrot ANAFI", "Parrot_Anafi/df077_Parrot_Anafi/", "Parrot_ ANAFI.pdf"),
    ("parrot_bebop_2", "Parrot Bebop 2 + SkyController", "Parrot_Bebop_2_plus_SkyController/df024_",
     "Parrot_Bebop_2.pdf"),
    ("yuneec_typhoon_h", "Yuneec Typhoon H", "Yuneec_Typhoon_H/df014_Yuneec_Typhoon_H/", "Yuneec_Typhoon_H.pdf"),
    ("yuneec_h520", "Yuneec H520", "Yuneec_H520/df033_Yuneec_H520/", "Yuneec_H520.pdf"),
    ("skydio_r1", "Skydio R1", "Skydio_R1/df051_Skydio_R1/", "Skydio_R1.pdf"),
    ("ryze_tello", "Ryze Tello", "Ryze_Tello/df046_Ryze_Tello/", "Ryze_Tello.pdf"),
    ("sensefly_ebee", "senseFly eBee", "SenseFly_eBee/df036_SenseFly_eBee/", "senseFly_eBee.pdf"),
    ("intel_falcon_8", "Intel Falcon 8+", "Intel_Falcon_8+/df073_Intel_Falcon_8+/", "Intel_Falcon_ 8+.pdf"),
    ("qysea_fifish_p3", "QYSEA FiFish P3 (underwater drone)", "Qysea_FiFish_P3/df070_Qysea_FiFish_P3/",
     "QYSEA_FIFISH_P3.pdf"),
    ("parrot_bluegrass", "Parrot Bluegrass", "Parrot_Bluegrass/df075_Parrot_Bluegrass/", "Parrot_ Bluegrass.pdf"),
)


def _re_escape(s: str) -> str:
    import re
    return re.escape(s)


_DRONE_CASES = tuple(
    Case(f"vto_{cid}", f"VTO drone forensics: {title}", 2018, "reference_device",
         f"A {title} drone set up and flown by VTO Labs in a controlled, geofenced environment. The data storage of "
         "the drone, its controller and the paired mobile devices were acquired. The drone's information sheet "
         "documents the salted flights (dates, GPS boundaries, time zone).",
         VTO,
         evidence=(RE(f"vto_{cid}_acq", f"{title}: drone, controller and mobile acquisitions", "iot", (DRONES,),
                      include=rf"^{_re_escape(folder)}|^Reports/VTO_drone_report_v\.1\.[01]_{_re_escape(report)}$",
                      exclude=r"\.(md5|sha1|MD5|SHA1)$|_small\.pdf$", answer_key=r"^Reports/", device=title,
                      transfer_gb=6.0, group="heavy", max_artifact_bytes=600_000_000, max_image_bytes=8_500_000_000),),
         tags=("drone", "uas", "iot", "reference", "multi-device"),
         publisher_notes="For their Drone Forensics Program, VTO purchased 79 drones: 30 drone models, ~3 of each "
                         "model. Each drone was setup and operated in a controlled, geofenced environment. Attempts "
                         "were then made to acquire and image the data storage areas on each drone, the controller, "
                         "connected mobile devices and computers. The drones were completely torn down and "
                         "disassembled to identify data storage areas. The drone information sheet (README) gives the "
                         f"salted-data dates, location and time zone for this {title}.",
         source_url="https://cfreds.nist.gov (Drone Data Set)")
    for cid, title, folder, report in _DRONE_MODELS)


# ============================================================================ Ali Hadi challenges (archive.org)
def _ahadi(cid, n, title, ident, notes, include="", password="", tags=(), group="light", gb=4.0, platform="windows",
           exclude=r"\.E01\.csv$|s4a-challenge4", cap=1_000_000_000):
    return Case(f"ahadi_{cid}", f"Ali Hadi DFIR challenge #{n}: {title}" if n else f"Ali Hadi DFIR case: {title}",
                2020, "challenge", notes.split("\n\n")[0][:600], AHADI,
                evidence=(RE(f"ahadi_{cid}_img", f"{title}: forensic image", platform, (f"ia:{ident}",),
                             include=include, exclude=exclude, password=password, transfer_gb=gb, group=group,
                             max_artifact_bytes=cap),),
                tags=("ahadi", platform, *tags), publisher_notes=notes,
                source_url=f"https://archive.org/details/{ident}")


_AHADI_CASES = (
    _ahadi("01_webserver", 1, "Web Server Case", "dfir-case1",
           "A company's web server has been breached through their website. Our team arrived just in time to take a "
           "forensic image of the running system and its memory for further analysis. Your job is to find out what "
           "happened: how the attacker got in, what was done on the server, and what was left behind.\n\n"
           "Evidence: forensic image of the Windows web server (E01) and a memory dump of the running system.",
           password="file:Password-Challenge1.txt", tags=("web-server", "intrusion", "memory"), gb=3.3),
    _ahadi("02_policy_violation", 2, "User Policy Violation Case", "4orensics.case2.7z",
           "A Windows forensic image for training on file carving, NTFS, registry analysis and Windows artifacts. A "
           "user is suspected of violating the organisation's acceptable-use policy; reconstruct the user's activity "
           "from the image.", tags=("policy-violation",), gb=6.7, group="heavy"),
    _ahadi("03_mystery_hacked", 3, "Mystery Hacked System", "windows8.1challenge3.001.7z",
           "One day an IT dept. guy/gal came to work to find a message (left by an attacker) written in a file on "
           "his/her system! Determine how the Windows 8.1 system was compromised and what the attacker did.",
           tags=("intrusion",), gb=2.5),
    _ahadi("04_stealthy_ads", 4, "Launching Attacks from Alternate Data Streams", "StealthyADS",
           "Malicious executables were hidden in, and run from, NTFS Alternate Data Streams. Find the hidden "
           "executables, how they were launched, and the traces their execution left.",
           tags=("ntfs", "ads", "malware"), gb=7.3),
    _ahadi("05_bsides_amman", 5, "BSides Amman 2021 (illegal activity)", "BSidesAmman21.E01",
           "A user accessed confidential files using the accounts \"joker\" and \"IEUser\" as part of an illegal-activity "
           "investigation. Reconstruct what each account did on the Windows system.",
           include=r"\.E01\.7z\.\d{3}$|README", tags=("insider",), gb=5.3),
    _ahadi("06_browser_policy", 6, "Browser Policy Violation Case", "browser-policy-violation",
           "Windows and web-browser forensic investigation of a suspected browser-use policy violation: establish "
           "which browsers were used, what was visited, searched and downloaded, and when.",
           tags=("browser", "policy-violation"), gb=6.3),
    _ahadi("07_sysinternals", 7, "SysInternals Case", "sysinternals-case",
           "The user downloaded what they thought was the SysInternals tool suite, double-clicked it, but the tools "
           "did not open. Determine what was actually downloaded and executed, and what it did to the system.",
           tags=("malware", "intrusion"), gb=7.8),
    _ahadi("08_ntfs_hidden", 8, "NTFS File System Case", "ntfs-hidden-files",
           "Locate five hidden data items concealed using NTFS file system capabilities.", tags=("ntfs", "anti-forensics"),
           gb=0.01),
    _ahadi("09_encrypt_them_all", 9, "Encrypt Them All Case", "anti-forensics-case-2",
           "Data on this system was protected with AES, BitLocker and asymmetric encryption. Find what was encrypted, "
           "with which methods, and recover (decrypt) the data where the evidence allows it.",
           tags=("encryption", "anti-forensics"), gb=7.9),
    _ahadi("10_meeting_location", 10, "Meeting Location Case", "anti-forensics-case-4",
           "The suspect, Max, used methods to hide his activity. Recover the hidden activity and locate the "
           "coordinates of the meeting he arranged.", tags=("anti-forensics", "geolocation"), gb=16.3, group="heavy"),
    _ahadi("11_where_admin", 11, "Where Did Administrator Go?", "where-did-administrator-go",
           "The domain administrator's files disappeared. Investigate the server for signs of compromise and "
           "establish what happened to the administrator's data.", tags=("intrusion", "server"), gb=35.2, group="heavy"),
    _ahadi("12_ransomcare_memory", 0, "Memory Forensics #01: RansomCare", "ransomcare-memdump1",
           "Analyze the memory dump of a Windows 10 system infected with the RansomCare ransomware: identify the "
           "malicious process, its origin, persistence and impact.", tags=("memory", "ransomware"), gb=0.7,
           platform="memory", cap=6_000_000_000),
    _ahadi("13_unallocated", 0, "Unallocated Disk Space #01", "unallocated-case1.7z",
           "A system was compromised, but only the unallocated disk space could be preserved as evidence. Recover "
           "what can be recovered from it and reconstruct the compromise.", tags=("carving", "unallocated"), gb=0.6,
           platform="mixed"),
)


EXT_CASES: tuple[Case, ...] = (
    # ======================================================================== Digital Corpora scenarios
    Case("nitroba", "Nitroba University harassment (network forensics)", 2008, "scenario",
         "A chemistry teacher at the fictional Nitroba State University received harassing e-mails that originate "
         "from a student dorm room with an open Wi-Fi router. A packet capture from the dorm's Ethernet port, the "
         "teacher's screenshots and the class roster are provided.",
         DC_ORIGIN,
         evidence=(RE("nitroba_capture", "Nitroba dorm-room Ethernet tap capture + victim screenshots", "network",
                      ("dc:scenarios/2008-nitroba/",), exclude=r"slides|solution",
                      answer_key=r"solution", device="Ethernet tap on dorm port 140.247.62.34", transfer_gb=0.06),),
         docs=(Doc("scenarios/2008-nitroba/Nitroba Harassment Lab scenarios.pdf", "brief"),
               Doc("scenarios/2008-nitroba/Nitroba Harassment Lab scenarios with Solution.pdf", "answer_key"),
               Doc("scenarios/2008-nitroba/nitroba-solution.pdf", "answer_key"),
               Doc("scenarios/2008-nitroba/slides.pdf", "answer_key")),
         questions=("Which student in Chemistry 109 sent the harassing e-mail, and what conclusive evidence in the "
                    "packet capture supports it?",),
         tags=("network", "pcap", "harassment", "email"),
         publisher_notes="""You are a security administrator at the prestigious (and fictional) Nitroba State University.

Nitroba's IT department received an email from Lily Tuckrige, a teacher in the Chemistry Department. Tuckrige has been receiving harassing emails and she suspects that they are being sent by a student in her class Chemistry 109, which she is teaching this summer. The email was received at Tuckridge's personal email account, lilytuckrige@yahoo.com. She took a screenshot of the web browser and sent it in.

The system administrator who received the complaint wrote back to Tuckridge that Nitroba needed the full headers of the email message. Tuckridge responded by clicking the "Full message headers" button in Yahoo Mail and sent in another screen shot, this one with mail headers.

The mail header shows that the mail message originated from the IP address 140.247.62.34, which is a Nitroba student dorm room. Three women share the dorm room. Nitroba provides an Ethernet connection in every dorm room but not Wi-Fi access, so one of the women's friends installed a Wi-Fi router in the room. There is no password on the Wi-Fi.

Because several email messages appear to come from the IP address, Nitroba decides to place a network sniffer on the ethernet port. All of the packets are logged. On Monday 7/21 Tuckridge received another harassing email. But this time instead of receiving it directly, the perpetrator sent it through a web-based service called "willselfdestruct.com." The website briefly shows the message to Tuckridge, and then the website reports that the "Message Has Been Destroyed."

You have been given the screen shots, the packets that were collected from the Ethernet tap, and the Chem 109 roster. Your job is to determine if one of the students in the class was responsible for the harassing email and to provide clear, conclusive evidence to support your conclusion.""",
         source_url="https://digitalcorpora.org/corpora/scenarios/nitroba-university-harassment-scenario/"),

    Case("m57_jean", "M57-Jean: leaked salary spreadsheet", 2008, "scenario",
         "M57.Biz, a small start-up: a confidential spreadsheet with the names and salaries of key employees was found "
         "posted to the comments section of a competitor's website. It came from the laptop of Jean, the CFO.",
         DC_ORIGIN,
         evidence=(RE("m57_jean_laptop", "Jean's laptop (Windows XP), EnCase E01/E02", "windows",
                      ("dc:drives/nps-2008-m57-jean/",), exclude=r"\.(key|ppt)$", answer_key=r"Solution",
                      device="Jean (CFO) laptop", transfer_gb=3.0),),
         tags=("windows", "exfiltration", "email", "single-device"),
         publisher_notes="The M57-Jean scenario is a single disk image scenario involving the exfiltration of corporate "
                         "documents from the laptop of a senior executive. The scenario involves a small start-up "
                         "company, M57.Biz. A few weeks into inception a confidential spreadsheet that contains the names "
                         "and salaries of the company's key employees was found posted to the \"comments\" section of one "
                         "of the firm's competitors. The spreadsheet came from the laptop of Jean, the company's CFO.\n\n"
                         "Jean says that she has no idea how the data left her laptop and that she must have been hacked.\n\n"
                         "You have been given a disk image of Jean's laptop. Your job is to figure out how the data was "
                         "stolen, or if Jean isn't as innocent as she claims.",
         questions=("How did the salary spreadsheet leave Jean's laptop?", "Was Jean hacked, or is she not as innocent as "
                    "she claims?"),
         source_url="https://digitalcorpora.org/corpora/scenarios/m57-jean/"),

    *(Case(f"nps2011_{cid}", f"NPS 2011 scenario: {title}", 2011, "scenario",
           f"In 2011 the Naval Postgraduate School created disk images for several scenarios. This is the {title} "
           "scenario.", DC_ORIGIN,
           evidence=(RE(f"nps2011_{cid}_disk", f"NPS 2011 {title} scenario disk (E01)", "windows",
                        (f"dc:scenarios/{prefix}/",), exclude=r"\.xml$", transfer_gb=gb, group="heavy"),),
           tags=("windows", "single-device", "nps", *tags),
           publisher_notes=f"In 2011 the Naval Postgraduate School created disk images for several scenarios. This is "
                           f"the {title} scenario: {story}",
           source_url="https://cfreds.nist.gov (NPS 2011 scenarios)")
      for cid, title, prefix, gb, tags, story in (
          ("weapon_deletion", "Weapon Deletion", "2011-nps-1weapondeletion", 37.1, ("weapons", "deletion"),
           "a computer used in connection with weapons, on which relevant files were deleted. Recover what was "
           "deleted and reconstruct the user's activity."),
          ("weapons_2", "Weapons 2", "2011-nps-2weapons", 21.4, ("weapons",),
           "the second weapons scenario. Establish what the computer's user did in relation to weapons and with whom."),
          ("drug_traffic", "Drug Traffic", "2011-nps-4drugtraffic", 19.5, ("drugs",),
           "a computer of a user suspected of drug trafficking. Find evidence of the trafficking activity, contacts "
           "and logistics."),
          ("control", "Control", "2011-nps-5control", 21.7, ("control",),
           "the control image of the series (normal use, no scripted criminal activity). Establish what the user did "
           "and whether anything suspicious is present."))),

    Case("ngdc_2012", "National Gallery DC 2012: art theft and defacement plot", 2012, "scenario",
         "The 2012 National Gallery DC scenario spans approximately 10 days and encompasses two intertwined story arcs "
         "centred on an employee of the National Gallery DC art gallery. Criminal plans for theft and defacement are "
         "discussed among the actors; when suspicious activity is reported, devices are seized and network traffic "
         "logs are requested.",
         "Digital Corpora (NPS / USMA West Point 2012)",
         evidence=(
             RE("ngdc_tracy_macbook", "Tracy's home computer (MacBook Air), final image 2012-07-16", "macos",
                ("dc:scenarios/2012-ngdc/tracy-home/",), include=r"16-final\.E0\d$|README", transfer_gb=5.8,
                group="heavy", device="Tracy's MacBook Air, shared with her daughter Terry"),
             RE("ngdc_tracy_external", "Tracy's external hard drive, final image 2012-07-16", "mixed",
                ("dc:scenarios/2012-ngdc/tracy-external/",), include=r"16-final\.E01$|README", transfer_gb=3.8,
                device="External drive used between home and work computers"),
             RE("ngdc_tracy_phone", "Tracy's phone, final logical image (tar) 2012-07-15", "ios",
                ("dc:scenarios/2012-ngdc/tracy-phone/",), include=r"15-final\.tar$|README", transfer_gb=0.8,
                device="Tracy's personal phone"),
             RE("ngdc_carry_phone", "Carry's phone, Cellebrite image 2012-07-15", "android",
                ("dc:scenarios/2012-ngdc/carry-phone/",), include=r"15-final\.zip$|README", transfer_gb=0.2,
                device="Carry's phone"),
             RE("ngdc_carry_tablet", "Carry's tablet, logical image (tar) 2012-07-16", "android",
                ("dc:scenarios/2012-ngdc/carry-tablet/",), include=r"16-final\.tar$|README", transfer_gb=0.8,
                device="Carry's tablet (catsumtwelve e-mail account)"),
             RE("ngdc_keylogger_mail", "Keylogger e-mails sent from Tracy's MacBook to Joe", "generic",
                ("dc:scenarios/2012-ngdc/email/",), transfer_gb=0.01),
             RE("ngdc_network", "NGDC work network captures (interior / exterior, SSLstrip)", "network",
                ("dc:scenarios/2012-ngdc/net/",), transfer_gb=0.6, max_artifact_bytes=300_000_000),
         ),
         docs=(Doc("scenarios/2012-ngdc/answers.zip", "answer_key", members=r"\.pdf$"),),
         tags=("macos", "ios", "android", "network", "email", "multi-device", "theft", "steganography"),
         publisher_notes="""The 2012 National Gallery DC scenario spans approximately 10 days and encompasses two distinct yet intertwined story arcs. The scenario is centered around an employee at the National Gallery DC Art Gallery. Criminal plans for both theft and defacement are discussed amongst actors during the scenario, and evidence may remain across the digital devices they used. The scenario is terminated upon suspicious activity being reported to law enforcement at which point certain devices are seized and network traffic logs are requested.

The seized evidence has been processed by the ingest team of the crime laboratory. You have been provided with: Carry's phone on 2012-07-15 (Cellebrite image); Carry's tablet on 2012-07-16 (physical E01 and logical tar); e-mail messages generated by spyware installed on Tracy's MacBook Air that were periodically e-mailed to Joe; Tracy's phone on 2012-07-15; Tracy's external hard drive; Tracy's home computer (MacBook Air, shared with her daughter Terry); and exterior / interior packet dumps of the NGDC work network. Interior packets represent traffic between workstations and an SSL-stripping middleman; exterior traffic is headed out of the NGDC network to the internet. On July 12 the technician inadvertently captured the OUTPUT of the tcpdump command rather than the raw packets.""",
         answer_notes="""Publisher's scenario story (Digital Corpora page): Alex, a wealthy businessman with Krasnovian ties, contacts Carry, a Krasnovian supporter in the US, seeking to embarrass America by defacing foreign art (belonging to Majavia) on display in the National Gallery in July; he sends Krasnovian militants posing as tourists, and Carry is to plan getting them into the museum with tools. Tracy, a supervisor at the National Gallery and an acquaintance of Carry, exchanges small data with Carry under the pretext of organizing a flash mob, for money. Separately, Tracy discusses with her brother Pat (a D.C. police detective) stealing specific items (stamps) from the Gallery; this correspondence is on her work computer, personal phone and home computer. Carry uses steganography and encryption and a tablet with the catsumtwelve e-mail account for dealings with Alex. Joe, Tracy's ex-husband, had installed a keylogger on the MacBook Air to monitor their daughter Terry; he discovers the theft conspiracy and turns Tracy in, which reveals the Tracy-Carry contact and leads to Carry's tablet and phone being seized, revealing the defacing plot.""",
         source_url="https://digitalcorpora.org/corpora/scenarios/national-gallery-dc-2012-attack/"),

    Case("tuck_2019", "Tuck scenario (macOS): attempt to join a terrorist organization", 2019, "scenario",
         "A macOS computer (2019) of a person suspected of trying to join a terrorist organization.",
         DC_ORIGIN,
         evidence=(RE("tuck_mac", "Tuck's macOS computer (E01, 5 segments)", "macos", ("dc:scenarios/2019-tuck/",),
                      transfer_gb=10.0, group="extended", device="Suspect Mac (macOS 2019)"),),
         tags=("macos", "terrorism", "single-device"),
         publisher_notes="2019 Tuck scenario (Digital Corpora): a scenario involving a person who tries to join a "
                         "terrorist organization. The evidence is a forensic image (EnCase E01, five segments) of the "
                         "person's macOS computer. Reconstruct the user's activity, communications and any steps "
                         "taken toward joining the organization.",
         source_url="https://downloads.digitalcorpora.org/corpora/scenarios/2019-tuck/"),

    Case("nps_casper_rw", "NPS 2009 Casper-RW: bootable Ubuntu USB persistence over weeks", 2009, "reference_device",
         "Four successive images of the casper-rw persistence file (ext3) of a bootable Ubuntu 8.10 USB flash drive, "
         "repeatedly booted and used over several weeks to browse US Government websites.",
         DC_ORIGIN,
         evidence=(RE("nps_casper_rw_gens", "casper-rw (ext3) generations 0-3", "linux", ("dc:drives/nps-2009-casper-rw/",),
                      transfer_gb=0.32),),
         tags=("linux", "usb", "ext3", "time-series"),
         publisher_notes="This directory contains multiple images of the casper-rw file from a 2GB flash USB drive "
                         "that was created with a bootable copy of Ubuntu 8.10. The device has been repeatedly booted "
                         "and used over the course of several weeks. Files from US Government websites have been "
                         "downloaded. (Four generations: gen0..gen3.)",
         source_url="https://digitalcorpora.s3.amazonaws.com/corpora/drives/nps-2009-casper-rw/narrative.txt"),

    Case("nps_domexusers", "NPS 2009 Domex users: IM and e-mail between three users", 2009, "reference_device",
         "A Windows XP SP3 system with two users (domexuser1, domexuser2) who communicate with a third user "
         "(domexuser3) via instant messaging and e-mail.",
         DC_ORIGIN,
         evidence=(RE("nps_domexusers_disk", "Windows XP SP3 system (redacted E01)", "windows",
                      ("dc:drives/nps-2009-domexusers/",), include=r"redacted\.E0\d$|narrative", transfer_gb=4.3),),
         tags=("windows", "im", "email"),
         publisher_notes="DOMEXUSERS (NTFS). This is a disk image of a Windows XP SP3 system that has two users, "
                         "domexuser1 and domexuser2, who communicate with a third user (domexuser3) via IM and email. "
                         "The redacted version (Microsoft Windows executables redacted) is used here.",
         source_url="https://digitalcorpora.s3.amazonaws.com/corpora/drives/nps-2009-domexusers/narrative.txt"),

    *(Case(f"nps2011_{cid}_android", f"NPS 2011 Android reference: {model}", 2011, "reference_device",
           f"{model} images taken by NPS across a series of experiments (baseline, Google account, and scripted "
           "activity), physical (MTD/mmcblk) and logical (ClockworkMod backup) acquisitions.", DC_ORIGIN,
           evidence=(RE(f"nps2011_{cid}_images", f"{model}: experiment images", "android",
                        (f"dc:mobile/2011-android/{folder}/",), include=include, transfer_gb=gb),),
           docs=(Doc("mobile/2011-android/README.txt", "brief"),), tags=("android", "nand", "yaffs2", "reference"),
           publisher_notes="This directory holds a corpora of images taken from Android devices (HTC Nexus One: Android "
                           "2.2.1, build FRG83; Samsung Nexus S: Android 2.3, build GRH55). With the exception of the "
                           "baseline image, both phones were synced to the npsforensics@gmail.com account. The "
                           "ClockworkMod ROM Manager application was used to perform backups that provide the logical "
                           "acquisition. Directories 2011-NexusX-Y are physical images and 2011-NexusX-Y-1 the logical "
                           "image of experiment Y (1 = baseline, 1.5 = after associating a Google account, 2..7 = "
                           "further scripted activity; see README).",
           source_url="https://digitalcorpora.s3.amazonaws.com/corpora/mobile/2011-android/README.txt")
      for cid, model, folder, include, gb in (
          ("nexus_one", "HTC Nexus One", "NexusOne", r"NexusOne-7(-1)?\.zip$|NexusOne-1\.5\.zip$|mtd", 0.6),
          ("nexus_s", "Samsung Nexus S", "NexusS-1", r"NexusS-(1\.5|4|7\.5)(-1)?\.zip$", 0.62))),

    *(Case(f"hickman_{sid}", f"Reference device: {title}", year, "reference_device",
           f"{title} populated by Joshua Hickman with a documented set of apps and user activity. The image-creation "
           "document lists what was done and when; it is the investigation context.", HICKMAN,
           evidence=(RE(f"{sid}_image", f"{title}: {kind}", "android", (f"dc:{path}",), include=include,
                        answer_key=r"solution", transfer_gb=gb, group=group, max_total_bytes=1_500_000_000),),
           docs=docs, tags=("android", "single-device", "reference"),
           publisher_notes=f"An image of {title} was created by Joshua Hickman: several popular applications were "
                           "populated with user data utilizing the capabilities of each individual app. The image "
                           "creation document (inside the distribution, saved under context/brief) lists the apps, "
                           "accounts and activity, with dates and times.",
           source_url="https://thebinaryhick.blog/public_images/")
      for sid, title, year, path, include, kind, gb, group, docs in (
          ("android_6", "Android 6 (Marshmallow)", 2018, "mobile/", r"^android_6\.zip$", "userdata image", 4.7,
           "heavy", (Doc("mobile/android_6_solutions.pdf", "answer_key"),)),
          ("android_7", "Android 7 (Nougat), LG Nexus 5X", 2019, "mobile/", r"^android_7\.tar\.gz$",
           "physical image (blk0_mmcblk0)", 6.5, "heavy", ()),
          ("android_8", "Android 8 (Oreo), LG Nexus 5X", 2019, "mobile/", r"^android_8\.tar\.gz$",
           "physical image (blk0_mmcblk0)", 5.7, "heavy", ()),
          ("android_11", "Android 11", 2021, "mobile/", r"^android_11\.zip$", "full file system", 10.9, "heavy", ()),
          ("android_12", "Android 12", 2022, "mobile/", r"^android_12\.zip$", "full file system", 41.9, "extended", ()),
      )),

    # ======================================================================== NIST CFReDS
    Case("nist_hacking_case", "NIST CFReDS Hacking Case: 'Mr. Evil' and the abandoned laptop", 2004, "scenario",
         "On 09/20/04 a Dell CPi notebook was found abandoned with a wireless PCMCIA card and a homemade 802.11b "
         "antenna. It is suspected of being used for hacking (intercepting traffic at Wi-Fi hotspots).",
         NIST,
         evidence=(RE("nist_hacking_laptop", "Dell Latitude CPi (serial VLQLW), split dd image", "windows",
                      tuple(f"https://cfreds-archive.nist.gov/images/hacking-dd/SCHARDT.00{i}" for i in range(1, 9))
                      + ("https://cfreds-archive.nist.gov/images/hacking-dd/SCHARDT.LOG",),
                      device="Abandoned Dell CPi notebook", transfer_gb=5.4),),
         docs=(Doc("https://cfreds-archive.nist.gov/Hacking_Case.html", "brief", rename="Hacking_Case.html"),
               Doc("https://cfreds-archive.nist.gov/images/TestAnswers.pdf", "answer_key")),
         tags=("windows", "hacking", "wireless", "single-device"),
         publisher_notes="On 09/20/04, a Dell CPi notebook computer, serial # VLQLW, was found abandoned along with a "
                         "wireless PCMCIA card and an external homemade 802.11b antennae. It is suspected that this "
                         "computer was used for hacking purposes, although cannot be tied to a hacking suspect, Greg "
                         "Schardt. Schardt also goes by the online nickname of \"Mr. Evil\" and some of his associates "
                         "have said that he would park his vehicle within range of Wireless Access Points (like "
                         "Starbucks and other T-Mobile Hotspots) where he would then intercept internet traffic, "
                         "attempting to get credit card numbers, usernames & passwords.\n\n"
                         "Find any hacking software, evidence of their use, and any data that might have been "
                         "generated. Attempt to tie the computer to the suspect, Greg Schardt.",
         questions=(
             "What is the image hash? Does the acquisition and verification hash match?",
             "What operating system was used on the computer?", "When was the install date?",
             "What is the timezone settings?", "Who is the registered owner?", "What is the computer account name?",
             "What is the primary domain name?", "When was the last recorded computer shutdown date/time?",
             "How many accounts are recorded (total number)?",
             "What is the account name of the user who mostly uses the computer?",
             "Who was the last user to logon to the computer?",
             "A search for the name of Greg Schardt reveals multiple hits. One of these proves that Greg Schardt is "
             "Mr. Evil and is also the administrator of this computer. What file is it? What software program does "
             "this file relate to?",
             "List the network cards used by this computer.",
             "This same file reports the IP address and MAC address of the computer. What are they?",
             "Which NIC card was used during the installation and set-up for LOOK@LAN?",
             "Find 6 installed programs that may be used for hacking.", "What is the SMTP email address for Mr. Evil?",
             "What are the NNTP (news server) settings for Mr. Evil?", "What two installed programs show this information?",
             "List 5 newsgroups that Mr. Evil has subscribed to.",
             "What are the user settings that were shown when the user was online and in a chat channel (mIRC)?",
             "List 3 IRC channels that the user of this computer accessed.",
             "Ethereal was installed. What is the name of the file that contains the intercepted data?",
             "What type of wireless computer was the victim (person who had his internet surfing recorded) using?",
             "What websites was the victim accessing?", "Search for the main user's web based email address. What is it?",
             "Yahoo mail saves copies of the email under what file name?",
             "How many executable files are in the recycle bin?", "Are these files really deleted?",
             "How many files are actually reported to be deleted by the file system?",
             "Perform an Anti-Virus check. Are there any viruses on the computer?"),
         source_url="https://cfreds-archive.nist.gov/Hacking_Case.html"),

    Case("nist_rhino_hunt", "Rhino Hunt (DFRWS 2005 Rodeo): illegal rhinoceros images", 2005, "challenge",
         "Look for images of a rhinoceros in a USB image and network traces.",
         NIST,
         evidence=(RE("rhino_hunt_data", "DFRWS 2005 Rodeo: USB image and network traces", "mixed",
                      ("https://cfreds-archive.nist.gov/dfrws/DFRWS2005-RODEO.zip",), transfer_gb=0.01),),
         docs=(Doc("https://cfreds-archive.nist.gov/dfrws/DFRWS2005-answers.pdf", "answer_key"),),
         tags=("usb", "pcap", "images", "carving"),
         publisher_notes="Rhino Hunt (from the DFRWS 2005 Rodeo): a city ordinance prohibits trafficking in "
                         "rhinoceros images. Look for images (of a rhinoceros) in an image file (a USB drive) and in "
                         "network traces, and determine how they were obtained and moved.",
         source_url="https://cfreds-archive.nist.gov/dfrws/Rhino_Hunt.html"),

    Case("nist_russian_tea_room", "Russian Tea Room: reconstruct a stolen menu (UTF-16 search)", 2007, "challenge",
         "Boris and Natasha stole the new menu of the Little Russian Tea Room; reconstruct it from an image of "
         "their hard drive.",
         NIST,
         evidence=(RE("russian_tea_room_disk", "Boris and Natasha's hard drive (EnCase E01)", "mixed",
                      ("https://cfreds-archive.nist.gov/001/CFReDS001.E01",), transfer_gb=0.01),),
         docs=(Doc("https://cfreds-archive.nist.gov/utf-16-russ.html", "brief", rename="utf-16-russ.html"),
               Doc("https://cfreds-archive.nist.gov/001/menu.pdf", "answer_key")),
         tags=("string-search", "unicode", "hpa"),
         publisher_notes="The evil Boris and Natasha have escaped from jail and are up to their old tricks. They have "
                         "stolen the new menu from The Little Russian Tea Room. Your task is to reconstruct the menu "
                         "from an image of their hard drive. To accomplish this you have to find all eight sections of "
                         "the menu located in the image (appetizers, soup, pancakes, meat pies and dumplings, meat and "
                         "fish, cheese and milk products, beverages, dessert). The text is bilingual English/Russian, "
                         "encoded in UTF-16BE.",
         source_url="https://cfreds-archive.nist.gov/utf-16-russ.html"),

    *_CFTT_CASES,

    # ======================================================================== DFRWS challenges
    Case("circl_wiped_disk", "CIRCL wiped disk: recovering data after wiping", 2023, "challenge",
         "A disk image that was wiped before seizure; the adversary interrupted the wiping, and the exercise is to "
         "establish what can still be recovered at byte level.",
         "CIRCL (via Digital Corpora)",
         evidence=(RE("circl_wiped", "Wiped disk (E01)", "mixed", ("dc:drives/circl-2023-wiped/",),
                      answer_key=r"wiped_disk\.pdf", transfer_gb=0.06),),
         tags=("anti-forensics", "wiping", "recovery"),
         publisher_notes="Recovering data from a wiped disk sounds impossible. But wiping a 'big' disk would take "
                         "time. In this CIRCL exercise an adversary started wiping the disk and was interrupted. "
                         "Analyze the byte-level data to recover plain files, file system structures or partitions "
                         "that survived.",
         source_url="https://digitalcorpora.org/circl-forensics-exercises/"),

    _dfrws(2005, "Professor Goatboy's laptop (Windows memory analysis)", ("windows", "memory", "intrusion", "ip-theft"),
           """For several years, Professor Goatboy has been performing secret research that is of great interest to a certain foreign government. In May 2005, rumors spread that he had written several papers detailing key aspects of his work but that he was being pressured not to publish them. To escape these pressures, the professor moved to a new research facility where he would be permitted to continue his work without interference.

In the last week of May, Professor Goatboy settled into his new office and moved his work onto the new laptop he had been assigned. Unfortunately, he was too busy during the first week at his new job to get much work done, and did not have time to secure the fresh installation of Windows 2000 on his laptop.

On Sunday June 5th, the research lab's incident response coordinator, Tom "Blackout Jack" Daniels, was examining network logs from the previous night and noticed unusual traffic coming from Professor Goatboy's computer. He promptly located the laptop in the professor's office, and used Helix 1.6 to dump physical memory. He attempted to find signs of intrusion on the system but had difficulty executing some of his tools. Specifically, the system would not run "pslist.exe" or "fport.exe" to gather information about running processes. When the system came back up, Daniels acquired the physical memory again, and again tried to acquire an image of the disk using Helix 1.6 under Windows without success. Finally, he rebooted the system using the Helix CD and acquired the drive using Grab 1.2.2.

The lab administration is seeking help in determining what occurred. In addition to the memory dumps, a file system timeline (Sleuthkit) and body file, the network capture log, ntoskrnl.exe and tcpip.sys from the system and the acquisition audit logs are available.""",
           evidence=(RE("dfrws2005_data", "Two physical memory dumps + timeline, body file, network log", "memory",
                        ("https://www.dropbox.com/s/cspavwi3w4wskuh/dfrws2005-physical-memory1.dmp.bz2?dl=0",
                         "https://www.dropbox.com/s/syr47y9nljpeguf/dfrws2005-physical-memory2.dmp.bz2?dl=0",
                         "gh:dfrws/dfrws2005-challenge/materials"), exclude=r"\.(exe|sys)(\.bz2)?$",
                        answer_key=DFRWS_ANSWERS, transfer_gb=0.1),),
           questions=("What hidden processes were running on the system, and how were they hidden?",
                      "What other evidence of the intrusion can be extracted from the memory dumps?",
                      "Why did pslist.exe and fport.exe not work on the compromised system?",
                      "Was the intruder specifically seeking Professor Goatboy's research materials?",
                      "Did the intruder obtain the Professor's research?",
                      "What computer was the intrusion launched from?",
                      "Is there any indication of who the intruder might be?")),

    _dfrws(2008, "Linux memory, disk and network fusion", ("linux", "memory", "pcap", "insider", "multi-source"),
           """Your organization has become aware of external attempts to gain access to sensitive proprietary information on its computer systems and has stepped up its monitoring in response. Data from this monitoring, in addition to interviews with employees, has focused attention on a single user, who is suspected of collaborating with an outside party.

When escalated monitoring of the user identified suspicious network and system activity, your organization's security team responded. They copied the contents of the user's home directory (this was a Linux desktop system), made a full dump of system memory and preserved a packet capture of network traffic from the system. It's your task to analyze this data and determine what can be established about the activity of the user.

The data set for this challenge contains files copied from the user's home directory, a dump of physical memory and a network packet capture in pcap format.""",
           evidence=(RE("dfrws2008_data", "Linux desktop: home directory copy, physical memory dump, packet capture",
                        "linux", ("gh:dfrws/dfrws2008-challenge/details",), answer_key=DFRWS_ANSWERS,
                        skip_docs=r"README\.md$", transfer_gb=0.1),),
           questions=("What relevant user activity can be reconstructed from the data and what does it show?",
                      "Is there evidence of inappropriate or suspicious activity on the system related to the user?",
                      "Is there evidence of collaboration with an outside party? If so, what can be determined about "
                      "the identity of the outside party? How was any collaboration conducted?",
                      "Is there evidence that sensitive data was copied? If so, what can be determined about that data "
                      "and the manner of transfer?")),

    _dfrws(2009, "PlayStation 3 'nssal' (Linux on PS3, memory, network)", ("linux", "ps3", "memory", "pcap", "illicit-images"),
           """In early 2009 it came to the attention of investigators that an individual with the nickname "nssal" was using a Sony Playstation3 (PS3) to make illicit images (specifically, certain images depicting Mardi Gras activities) available to other PS3 users. Investigators determined that "nssal" was connecting from an IP address in New Orleans, and they began capturing network traffic with the goal of catching "nssal" red-handed. Based on their initial surveillance, it appeared that "nssal" had advanced knowledge of Linux and digital forensics.

On March 11, 2009 investigators observed "nssal" communicating with another PS3 user and exchanging unknown data. With proper legal authorization, the investigators entered the suspect's premises and found him in front of a PS3 that was running Linux. They interviewed the suspect and determined that he was a digital forensics researcher who was developing memory acquisition and analysis tools for Linux on the PS3. He denied having exchanged any information with other PS3 users.

Investigators captured physical memory of the Linux system on the PS3 (nssal-physicalmem.dd), a forensic duplicate of the Linux partition on the PS3 (nssal-linux-side-fs.dd) and the suspect's thumbdrive (nssal-thumb-fs.dd). Network traces: nssal-capture-1.pcap (early surveillance), nssal-capture-2.pcap (communication between "nssal" and a machine at Johns Hopkins University), plus the Johns Hopkins side (jhuisi-capture-1.pcap, jhuisi-linux-side-fs.dd).""",
           evidence=(RE("dfrws2009_data", "PS3 memory, PS3 Linux partition, thumb drive, network traces", "linux",
                        ("gh:dfrws/dfrws2009-challenge/materials",
                         "https://www.dropbox.com/s/7y34kulkazm20ko/jhuisi-capture-1.pcap.bz2?dl=0",
                         "https://www.dropbox.com/s/ao0h0ptdi1xfz31/jhuisi-linux-side-fs.dd.bz2?dl=0",
                         "https://www.dropbox.com/s/q4yv6omhaypigi9/nssal-physicalmem.dd.bz2?dl=0",
                         "https://www.dropbox.com/s/s7v9w0n4c1htge3/nssal-linux-side-fs.dd.bz2?dl=0"),
                        answer_key=DFRWS_ANSWERS, transfer_gb=1.0),),
           questions=("What relevant user activity can be reconstructed from the available forensic data and what does it show?",
                      "Is there evidence of inappropriate or suspicious activity on the system?",
                      "Is there evidence of collaboration with an outside party? If so, what can be determined about the "
                      "identity of the outside party? How was any collaboration conducted?",
                      "Is there evidence that illicit data (specifically, Mardi Gras images) was exchanged? If so, what can "
                      "be determined about that data and the manner of transfer?",
                      "What data (if any) was provided by the Johns Hopkins PS3?",
                      "The suspect claims that he was not responsible for any transfer of data. What evidence do you have "
                      "to show that remote, unauthorized access to the system might have occurred, and does this "
                      "evidence exonerate the suspect?")),

    _dfrws(2010, "'The General': arms dealer's Sony Ericsson K800i", ("feature-phone", "nand", "nor", "arms-trafficking"),
           """After an extensive undercover operation, a known arms dealer named Monsieur Victor, commonly known as "The General," was lured out of hiding and apprehended in the Netherlands. He had expected a meeting to finalize a large sale of weapons, including tanks, missiles, attack helicopters, and assault rifles. Instead he met with police. When he realized the situations, he threw a mobile device in a nearby canal. The device was later retrieved by scuba divers, and was found to be a Sony Ericsson K800i Cybershot.

Mr. Victor has been connected with several front companies, and a fleet of military cargo planes used to deliver weaponry. He has openly mocked arms embargoes, and is suspected of selling arms to both sides in military conflicts around the globe. In the past, investigators could not find sufficient evidence to link him to any arms deals.

In the current operation, undercover investigators arranged an arms deal through one of Mr. Victor's front companies, Smurf Celtic. To convince him that the deal was real, an initial down payment was made by electronic funds transfer to Smelt Bank in France into an account owned by another front company named RipTide Security. His meeting in Amsterdam was to finalize the full payment to a bank account in Dubai.

After Mr. Victor's Sony Ericsson K800i Cybershot was retrieved, it was provided to the Netherlands Forensic Institute for processing. The Memory toolkit was used to acquire the contents of NAND and NOR Flash memory from the device.

You have been asked to recover any evidence that can connect Monsieur Victor to the sale of arms through Smurf Celtic, and the receipt of payment to RipTide Security. In addition, you have been asked to recover any leads that might connect him to other individuals, companies, or bank accounts that are involved in Mr. Victor's international arms business.""",
           evidence=(RE("dfrws2010_k800i", "Sony Ericsson K800i NAND + NOR flash dumps (with audit logs)", "generic",
                        ("https://www.dropbox.com/s/wybw2kim5ikimqb/dfrws2010-challenge.zip?dl=0",),
                        transfer_gb=0.04, max_artifact_bytes=300_000_000),),
           questions=("What evidence connects Monsieur Victor to the sale of arms through Smurf Celtic?",
                      "What evidence shows the receipt of payment to RipTide Security?",
                      "What leads connect him to other individuals, companies or bank accounts in his arms business?")),

    _dfrws(2011, "Android scenario 1: suspicious death of Donald Norby", ("android", "yaffs2", "death-investigation"),
           """Donald Norby was found dead in his home with a single bullet to the head. It is unclear whether this is a suicide or homicide. The largest question revolves around the victim's potential connections to an organized criminal group called KRYPTIX. You have been asked to perform a forensic examination of Norby's Android device found at the scene in order to determine his activities and, possibly, who he communicated with prior to his death. Your ultimate goal is to determine whether he killed himself or was murdered and provide any further leads to the investigator.

The device was acquired using what the agent considered to be industry best practices. The device flash storage as well as removable media was collected. See the case specific logs for more information.""",
           evidence=(RE("dfrws2011_case1_phone", "Norby's Android phone: flash partitions + removable media", "android",
                        ("https://www.dropbox.com/s/fzfb7uabq6dlb51/Case1.tgz?dl=0",), transfer_gb=0.16, group="heavy"),),
           case_id="dfrws2011_case1",
           questions=("Did Donald Norby kill himself or was he murdered?",
                      "What were his activities and who did he communicate with prior to his death?",
                      "Was the device rooted? Did it have adb enabled? Was the collection process sound?",
                      "What applications were installed (with versions)?")),
    _dfrws(2011, "Android scenario 2: SwiftLogic intellectual property theft", ("android", "yaffs2", "ip-theft"),
           """A serious breach of security occurred within an organization named SwiftLogic Inc.; valuable documents containing designs of a new product named Palomino were confirmed to have been leaked to a competitor. Based on an internal investigation, Yob Taog is suspected of the leak and was suspended pending investigation. Taog's Android smartphone was - surprisingly - voluntarily submitted by Taog for further investigation. Your goal in this investigation is to document any evidence that intellectual property was stolen, to support termination of Yob Taog and potential criminal charges. You are asked to perform a forensic analysis of his device for evidence of the breach.

The device was acquired using what the agent considered to be industry best practices. The device flash storage as well as removable media was collected. See the case specific logs for more information.""",
           evidence=(RE("dfrws2011_case2_phone", "Yob Taog's Android phone: flash partitions + removable media", "android",
                        ("https://www.dropbox.com/s/zaf541kv581tfcv/Case2.tgz?dl=0",), transfer_gb=0.34, group="heavy"),),
           case_id="dfrws2011_case2",
           questions=("Is there evidence that intellectual property (Palomino designs) was stolen, and how?",
                      "Was the device rooted? Did it have adb enabled? Was the collection process sound?",
                      "What applications were installed (with versions)?")),

    _dfrws(2016, "Software-defined networking switch (memory + southbound traffic)", ("network", "sdn", "memory", "pcap"),
           """You have obtained a memory image from an SDN switch, along with a capture of the network traffic between the SDN switch and its controller(s), i.e., the "southbound" traffic. Your job is to analyze these artifacts for forensically interesting information.

We know what hosts were connected to the SDN switch, what they did, and who they communicated with -- as well as the network policy and configuration as implemented by our SDN controller. Reverse engineer as many of these details as possible.""",
           evidence=(RE("dfrws2016_switch", "SDN switch memory image + southbound packet capture", "network",
                        ("https://www.dropbox.com/s/2eyf3rj4bxwdlq0/ram.raw.bz2?dl=0",
                         "gh:dfrws/dfrws2016-challenge/materials"), answer_key=DFRWS_ANSWERS, transfer_gb=0.5,
                        max_artifact_bytes=2_200_000_000),),
           questions=("What type of SDN switch and controller are in use?",
                      "What hosts (MAC or IP) were connected to which switch ports?",
                      "How much traffic did these hosts send, and to whom?", "When were these hosts active?",
                      "What flows does the SDN switch match on? Are they static or dynamic rules?",
                      "What actions does the SDN switch take, and for which flow rules?")),

    _dfrws(2017, "IoT: the murder of Betty (smart home)", ("iot", "android", "smart-home", "homicide", "multi-device"),
           """A woman (Betty) has been murdered. The murder was called in by Betty's husband (Simon), who claims to have been at home at the time. Additional case details, including a report from the scene with an interview of one of the suspects, are in the IoT Forensic Challenge Case Details document. Your job is to analyze these artifacts for forensically interesting information. Time Zone: UTC+9.

Responders acquired: a Smart TV Raspberry Pi, Betty's Samsung Note 2, Simon's Samsung Note 2, a Google OnHub diagnostic report, Amazon Echo cloud data, and the smart-home network dump.""",
           evidence=(RE("dfrws2017_iot", "Smart TV (Raspberry Pi), two Samsung Note 2 phones, OnHub, Echo cloud, network dump",
                        "mixed", ("https://www.dropbox.com/s/vvkza954gn9sx89/001-SmartTV-RaspberryPi.zip?dl=0",
                                  "https://www.dropbox.com/s/216qsdvughbp9n7/002-BettyNote2Black.zip?dl=0",
                                  "https://www.dropbox.com/s/wl28k99ya27dm13/003-SimonNote2White.zip?dl=0",
                                  "gh:dfrws/dfrws2017-challenge"),
                        include=r"^(001|002|003)-|^materials/|IoT Forensic Challenge Case Details\.pdf$|challenge-details",
                        answer_key=DFRWS_ANSWERS, transfer_gb=10.0, group="heavy"),),
           questions=("Assign the probability of the results given two competing propositions (the husband killed the "
                      "wife; some unknown person did).", "What do the IoT, phone, network and cloud traces show about "
                      "the time of the murder and who was present?")),

    _dfrws(2018, "IoT: raid on Jessie Pinkman's drug lab", ("iot", "android", "smart-home", "drugs", "multi-device"),
           """On 17 May 2018 at 10:40, the police were alerted that an illegal drug lab was invaded and unsuccessfully set on fire. The police respond promptly, and a forensic team is on scene at 10:45, including a digital forensic specialist.

The owner of the illegal drug lab, Jessie Pinkman, is nowhere to be found. Police interrogate two of Jessie Pinkman's known associates: D. Pandana and S. Varga. Pandana and Varga admit having access to the drug lab's WiFi network but deny any involvement in the raid. They also say that Jessie Pinkman had the IoT security systems installed because he feared attacks from a rival gang and that Jessie kept the alarm engaged in "Home" mode whenever he was inside the drug lab.

Within the drug lab the digital forensic specialist observes some IoT devices, including an alarm system (iSmartAlarm), three cameras (QBee Camera, Nest Camera and Arlo Pro) as well as a smoke detector (Nest Protect). An Amazon Echo and a WinkHub are also present. The specialist preserves the diagnostic logs from the iSmartAlarm base station, and acquires a copy of the filesystem of the WinkHub. He also collects the iSmartAlarm and Arlo base stations. The QBee Camera seems to be disabled, so he collects a sample of the network traffic. Back at the laboratory, memory images of the two base stations and an archive of the Arlo NAND folder are collected. Jessie Pinkman's Samsung Galaxy S6 Edge is found at the scene; a physical image is acquired. Time Zone: UTC+2.""",
           evidence=(RE("dfrws2018_iot", "iSmartAlarm + Arlo memory/NAND, WinkHub FS, Echo cloud, network capture, Galaxy S6 Edge",
                        "mixed", ("gh:dfrws/dfrws2018-challenge/materials",
                                  "https://www.dropbox.com/s/kpvrrmn26ljo64g/dfrws_arlo.img?dl=0",
                                  "https://www.dropbox.com/s/ww758tm15yisg8a/Samsung%20GSM_SM-G925F%20Galaxy%20S6%20Edge.7z?dl=0"),
                        answer_key=DFRWS_ANSWERS, transfer_gb=6.0, group="heavy"),),
           questions=("At what time was the illegal drug lab raided?",
                      "Could any of the two friends of Jessie Pinkman have been involved in the raid? If yes, which "
                      "friend, and with what confidence?", "How was the QBee camera disabled?")),

    _dfrws(2023, "The Troubled Elevator (ICS / PLC)", ("ics", "plc", "memory", "pcap", "cctv"),
           """Kristi Wayne from Wayne Enterprise has recently bought a controversial bank in the city of Richmond.

On June 29, Friday afternoon, during her visit to the bank, she used an executive-only elevator designed to provide a smooth and private commute for the high-ranking officials within the bank. Wayne enters the elevator and presses the button to get to another floor. However, the elevator suddenly starts malfunctioning, trapping Wayne inside. Wayne calls from the elevator for emergency assistance. After an extended episode of patience and misery, she is finally rescued. Due to this high-profile incident, your forensic team has been called for an investigation. Fortunately, the elevator infrastructure is designed to log network traffic and device memory dumps for a certain time period.

Available: the M221 PLC elevator programming manual and network diagrams, a network trace, a RAM dump of the CEO's desktop, a series of PLC external / on-chip RAM dumps, and CCTV footage.""",
           evidence=(RE("dfrws2023_elevator", "Network trace, CEO desktop RAM, PLC RAM dumps, CCTV footage", "mixed",
                        ("gh:dfrws/dfrws2023-challenge",), answer_key=DFRWS_ANSWERS, skip_docs=r"README\.md$",
                        transfer_gb=1.3, max_artifact_bytes=1_500_000_000),),
           questions=("What were the elevator's behaviors during the malfunction?", "What is the timeline of the malfunction?",
                      "What was the specific cause of the malfunction?", "Is there any evidence of an inside attacker?",
                      "What attack evidence exists on the network, the computer and the PLC device?")),

    # ======================================================================== Ali Hadi
    *_AHADI_CASES,

    # ======================================================================== DFIR Madness
    Case("dfirmadness_case001", "DFIR Madness Case 001: The Stolen Szechuan Sauce", 2020, "challenge",
         "A mad-scientist boss learns from the FBI that his secret Szechuan sauce recipe is on the dark web. A domain "
         "controller and a desktop were imaged, with memory, autoruns and a network capture.",
         "DFIR Madness (James Smith)",
         evidence=(
             RE("szechuan_dc01", "DC01 (domain controller): disk E01, memory, pagefile, autoruns, protected files",
                "windows", tuple(f"https://dfirmadness.com/case001/{n}" for n in (
                    "DC01-E01.zip", "DC01-memory.zip", "DC01-autorunsc.zip", "DC01-ProtectedFiles.zip")),
                transfer_gb=5.5, max_artifact_bytes=2_200_000_000),
             RE("szechuan_desktop", "DESKTOP-SDN1RPT: disk E01, memory, autoruns, protected files", "windows",
                tuple(f"https://dfirmadness.com/case001/{n}" for n in (
                    "DESKTOP-E01.zip", "DESKTOP-SDN1RPT-memory.zip", "DESKTOP-SDN1RPT-autorunsc.zip",
                    "DESKTOP-SDN1RPT-Protected%20Files.zip")), transfer_gb=7.5, max_artifact_bytes=2_200_000_000),
             RE("szechuan_pcap", "Case 001 network capture", "network",
                ("https://dfirmadness.com/case001/case001-pcap.zip",), transfer_gb=0.15),
         ),
         tags=("windows", "active-directory", "intrusion", "memory", "pcap", "multi-device"),
         publisher_notes="Your bedroom door bursts open, shattering your pleasant dreams. Your mad scientist of a boss "
                         "begins dragging you out of bed by the ankle. He simultaneously explains between belches that "
                         "the FBI contacted him. They found his recently-developed Szechuan sauce recipe on the dark web. "
                         "Determine how the breach occurred, identify the malware and attacker infrastructure, and what "
                         "was stolen. Note: this incident occurred at an organization located in Colorado in "
                         "September, so the incident is at UTC-6.",
         questions=("What's the operating system of the server? Of the desktop?", "What was the local time of the server?",
                    "Was there a breach?", "What was the initial entry vector (how did they get in)?",
                    "Was malware used? If so, what was it (process name, source IP, C2 IP, location on disk, when it "
                    "first appeared, capabilities, persistence)?",
                    "What malicious IP addresses were involved, and are any of them known?",
                    "What other systems were accessed, and what data was stolen from them (and when)?",
                    "What was the network layout of the victim network and what architecture changes are needed?",
                    "When was the Szechuan sauce recipe stolen?", "What other sensitive files were accessed?",
                    "When was the last known contact with the adversary?"),
         source_url="https://dfirmadness.com/the-stolen-szechuan-sauce/"),

    # ======================================================================== Hexordia / Magnet Virtual Summit / MSAB
    Case("mvs_2023", "Magnet Virtual Summit 2023 CTF", 2023, "ctf",
         "Magnet Virtual Summit 2023 CTF evidence: a Google Pixel 3a XL logical image, a Windows PC image and an "
         "iPhone full file system.",
         "Magnet Forensics / Hexordia (via NIST CFReDS)",
         evidence=(
             RE("mvs2023_pixel", "Google Pixel 3a XL logical image (tar)", "android",
                ("https://storage.googleapis.com/mvs-2023/Google%20Pixel%203a%20XL%20Logical%20Image%20-%20Data.tar",),
                transfer_gb=9.8, group="heavy"),
             RE("mvs2023_iphone", "iPhone full file system (zip)", "ios",
                ("https://storage.googleapis.com/mvs-2023/00008101-0010541A1130001E_files_full-001.zip",),
                transfer_gb=1.5, group="heavy"),
             RE("mvs2023_pc", "Windows PC PC-MUS-001 (E01)", "windows",
                ("https://storage.googleapis.com/mvs-2023/PC-MUS-001.E01",), transfer_gb=6.0, group="extended"),
         ),
         tags=("android", "ios", "windows", "multi-device", "ctf"),
         publisher_notes="Magnet Virtual Summit 2023 Capture the Flag: three evidence items from the same scenario "
                         "(a Google Pixel 3a XL logical extraction, an iPhone full file system extraction and a "
                         "Windows PC image, PC-MUS-001). Questions were released during the event.",
         source_url="https://cfreds.nist.gov (Magnet Virtual Summit 2023)"),

    Case("mvs_2024", "Magnet Virtual Summit 2024 CTF: assault at Ethan Allen Park", 2023, "ctf",
         "Chadwick alleges that Rocco assaulted him on 17 Dec 2023 at Ethan Allen Park, Burlington VT, during a "
         "meeting to buy 'vitamins'. Rocco's Android Pixel 3 is provided.",
         "Hexordia (Magnet Virtual Summit 2024)",
         evidence=(RE("mvs2024_pixel3", "Rocco Sachs's Android Pixel 3 (logical tar)", "android",
                      ("https://drive.google.com/file/d/1S6btpUWNOBqAwCkk-XTqFA8R7oNoAaSa/view",),
                      transfer_gb=27.4, group="heavy", device="Android Pixel 3 of Rocco Sachs"),),
         tags=("android", "assault", "drugs", "ctf"),
         publisher_notes="Chadwick alleges that Rocco assaulted him on the evening of December 17, 2023, at the Ethan "
                         "Allen Park in Burlington VT. Two phones have been received: Android Pixel 3 of Rocco Sachs; "
                         "iPhone 14 of Chadwick (Chad) Elms.\n\nOn Dec 16, 2023, Chadwick and Rocco connected on X, "
                         "formerly known as Twitter. On Dec 17, 2023, Chadwick and Rocco begin playing Call of Duty "
                         "together and chatting in the game. They have conversations about \"vitamins\" on X and then "
                         "move the conversation to Text. Via text, Chadwick requests a sample and they plan to meet at a "
                         "Burlington City Hall park at noon. They plan to meet again later that evening, 9pm, at Ethan "
                         "Allen Park, to exchange more cash for vitamins. (Publisher's scenario summary; the evidence "
                         "published with it is Rocco's Pixel 3.)",
         answer_notes="Publisher's scenario continues: at the noon meeting Chadwick places an AirTag on Rocco's person "
                      "unbeknownst to him. At the 9pm meeting at Ethan Allen Park an altercation occurs: Rocco takes the "
                      "money and assaults Chadwick. On Dec 18, 2023 both travel out of the area.",
         source_url="https://cfreds.nist.gov (2024 MVS CTF)"),

    Case("mvs_2026", "Magnet Virtual Summit 2026 CTF (Android, iOS, Mac)", 2025, "ctf",
         "Hexordia's Magnet Virtual Summit 2026 CTF: a Samsung Galaxy A53, an iPhone 14 Plus full file system and "
         "a MacBook Pro image.",
         "Hexordia (Magnet Virtual Summit 2026)",
         evidence=(
             RE("mvs2026_samsung_a53", "Samsung Galaxy A53 extraction", "android",
                ("https://mvs26.hexordiactf.com/SamsungA53.zip",), transfer_gb=22.0, group="extended"),
             RE("mvs2026_iphone14plus", "iPhone 14 Plus full file system", "ios",
                ("https://mvs26.hexordiactf.com/iPhone14Plus.zip",), exclude=r"keychain", transfer_gb=20.0,
                group="extended"),
             RE("mvs2026_macbook", "MacBook Pro image", "macos", ("https://mvs26.hexordiactf.com/MacBookPro.zip",),
                transfer_gb=42.5, group="extended"),
         ),
         docs=(Doc("https://www.hexordia.com/blog/magnet-virtual-summit-2026-ctf-android-questions", "answer_key",
                   rename="mvs2026_android_questions_and_answers.html"),
               Doc("https://www.hexordia.com/blog/magnet-virtual-summit-2026-ctf-ios-questions", "answer_key",
                   rename="mvs2026_ios_questions_and_answers.html"),
               Doc("https://www.hexordia.com/blog/magnet-virtual-summit-2026-ctf-mac-questions", "answer_key",
                   rename="mvs2026_mac_questions_and_answers.html")),
         questions=("Anything open this time of night? Name of the establishment in the screenshot.",
                    "When was Clash of Clans last closed (UTC)?", "From which country did the rejected/declined call originate?",
                    "What is the ingress rating of the night-vision security camera?",
                    "What was the answer to the first Wordle puzzle solved?",
                    "What is the name of the most recently worked-on client?",
                    "Why was the Nike app last exited?", "Which swimming class has the most badges?",
                    "What was the second guess on the first Wordle played?",
                    "How many e-mails were auto-labelled as notifications?",
                    "What is the hex colour of the content expiring 2025-12-01 15:06:45?",
                    "What is the PID of the app in focus at 2025-12-15 17:48:09?",
                    "When (UTC) was the time zone set?", "What is the top package name in the split-screen shortcut?"),
         tags=("android", "ios", "macos", "multi-device", "ctf"),
         publisher_notes="Magnet Virtual Summit 2026 CTF images created by Hexordia: Samsung Galaxy A53 (Android), "
                         "iPhone 14 Plus (iOS full file system) and MacBook Pro (macOS). The questions (listed with "
                         "this case) cover app usage, communications, games, location and system configuration.",
         source_url="https://www.hexordia.com/blog/magnet-virtual-summit-2026-ctf-android-questions"),

    Case("msab_ctf_2026", "MSAB 2026 CTF (Android, iOS)", 2026, "ctf",
         "MSAB 2026 CTF evidence created by Hexordia: a Samsung Galaxy S20 and an iPhone 12.",
         "MSAB / Hexordia",
         evidence=(
             RE("msab2026_s20", "Samsung Galaxy S20 extraction", "android", ("https://msab26.hexordiactf.com/SamsungS20.zip",),
                transfer_gb=39.4, group="extended"),
             RE("msab2026_iphone12", "iPhone 12 extraction", "ios", ("https://msab26.hexordiactf.com/iPhone12.zip",),
                exclude=r"keychain", transfer_gb=50.1, group="extended"),
         ),
         docs=(Doc("https://www.hexordia.com/blog/2026-msab-ctf-android-questions", "answer_key",
                   rename="msab2026_android_questions_and_answers.html"),
               Doc("https://www.hexordia.com/blog/2026-msab-ctf-ios-questions", "answer_key",
                   rename="msab2026_ios_questions_and_answers.html")),
         tags=("android", "ios", "multi-device", "ctf"),
         publisher_notes="2026 MSAB CTF: a Samsung Galaxy S20 (Android) and an iPhone 12 (iOS) populated by Hexordia "
                         "for the MSAB capture-the-flag event. The examiner answers questions about the devices' "
                         "users, communications, app activity and locations.",
         source_url="https://www.hexordia.com/blog/2026-msab-ctf-android-questions"),

    # ======================================================================== Cellebrite CTFs
    Case("cellebrite_ctf_2021", "Cellebrite CTF 2021: Heisenberg, Marsha and Beth", 2021, "ctf",
         "Cellebrite's 2021 capture-the-flag: a Samsung Note 10 (Heisenberg), two iPhone X full file systems "
         "(Marsha, Beth) and Marsha's PC.",
         "Cellebrite (via NIST CFReDS)",
         evidence=(
             RE("cb2021_heisenberg", "Heisenberg's Samsung SM-N970U1 (Qualcomm Live)", "android",
                ("https://drive.google.com/drive/folders/1aPuq-c1U1txOBJQIA688u6bFzdz59FJ6",),
                include=r"Heisenberg|Walk-Through|ZIP_Password", password="file:ZIP_Password.txt",
                answer_key=r"Walk-Through|Wrap", transfer_gb=14.0, group="heavy"),
             RE("cb2021_beth_iphone", "Beth's iPhone X (FFS, checkm8)", "ios",
                ("https://drive.google.com/drive/folders/1aPuq-c1U1txOBJQIA688u6bFzdz59FJ6",),
                include=r"Beth_iPhone|ZIP_Password", password="file:ZIP_Password.txt", transfer_gb=8.9, group="heavy"),
             RE("cb2021_marsha_iphone", "Marsha's iPhone X (FFS Premium, split zip)", "ios",
                ("https://drive.google.com/drive/folders/1aPuq-c1U1txOBJQIA688u6bFzdz59FJ6",),
                include=r"Marsha_iPhoneX|ZIP_Password", password="file:ZIP_Password.txt", transfer_gb=29.8,
                group="extended"),
             RE("cb2021_marsha_pc", "Marsha's PC (physical, split zip)", "windows",
                ("https://drive.google.com/drive/folders/1aPuq-c1U1txOBJQIA688u6bFzdz59FJ6",),
                include=r"Marsha_PC|ZIP_Password", password="file:ZIP_Password.txt", transfer_gb=46.6,
                group="extended"),
         ),
         tags=("android", "ios", "windows", "multi-device", "ctf", "drugs"),
         publisher_notes="Cellebrite's 2021 Capture the Flag published four extractions of persons in the same "
                         "storyline: Heisenberg's Samsung SM-N970U1 (Qualcomm Live extraction, 2021-07-22), Beth's iPhone "
                         "X (full file system via checkm8, 2021-07-29), Marsha's iPhone X (full file system, Premium) "
                         "and Marsha's PC (physical). The examiner answers the CTF questions about these persons' "
                         "activity. Cellebrite later published its own explanations; they are kept apart as the answer key.",
         source_url="https://cfreds.nist.gov (Cellebrite 2021 CTF)"),

    Case("cellebrite_ctf_2023", "Cellebrite CTF 2023: Sharon, Abe, Russell and Felix", 2023, "ctf",
         "Cellebrite's 2023 capture-the-flag: full file system extractions of four persons' iOS and Android devices.",
         "Cellebrite (via NIST CFReDS)",
         evidence=tuple(
             RE(f"cb2023_{who.lower()}", f"{who}'s device (FFS extraction)", "mixed", (url,), password="M0@r_c0WB3L1!!",
                transfer_gb=gb, group="heavy" if gb < 20 else "extended")
             for who, url, gb in (
                 ("Felix", "https://drive.google.com/file/d/1zV9Eon62tTzx6LGytx32Ruzuw6H_Xs07/view", 9.7),
                 ("Sharon", "https://drive.google.com/file/d/1BsTuhweokILexwh-qnTE1H7R-bS8dAOn/view", 18.2),
                 ("Russell", "https://drive.google.com/file/d/1Ly69KLYN8H2kc4OGoJc2VgZzkxAXObpx/view", 25.0),
                 ("Abe", "https://drive.google.com/file/d/1dn5IRU1Sa3A9NokW5v5HhJsiRud3tYHa/view", 26.0))),
         tags=("android", "ios", "multi-device", "ctf"),
         publisher_notes="Cellebrite CTF 2023: 4 datasets (Sharon, Abe, Russell & Felix): full file system "
                         "extractions from iOS and Android devices of four persons in one storyline. The examiner "
                         "answers the CTF questions about their communications and activity.",
         source_url="https://cfreds.nist.gov (Cellebrite's Capture the Flag 2023)"),

    Case("cellebrite_ctf_2024", "Cellebrite CTF 2024: Otto, Felix, Russell and Sharon", 2024, "ctf",
         "Cellebrite's 2024 capture-the-flag: extractions associated with four suspects, with a backstory document.",
         "Cellebrite (via NIST CFReDS)",
         evidence=(
             RE("cb2024_file1", "Sharon and Felix extractions (+ case PDF)", "mixed",
                ("https://drive.google.com/file/d/1lONcY9BZ6srfvrQX6uGe-AXdK-6K0nLz/view",),
                exclude=r"CLBE", password="V@nD0wnByTh3R1veR", transfer_gb=60.0, group="extended"),
             RE("cb2024_otto", "Otto's extraction", "mixed",
                ("https://drive.google.com/file/d/1PrGE-Z6jZTZzPPzAdQoQI3SHDE3fM0ku/view",),
                password="V@nD0wnByTh3R1veR", transfer_gb=43.5, group="extended"),
         ),
         tags=("android", "ios", "multi-device", "ctf"),
         publisher_notes="Cellebrite CTF 2024: datasets associated with each suspect (Otto, Felix, Russell, Sharon). "
                         "The backstory and questions are in the Cellebrite notebook (CellebriteCTF2024.pdf, saved "
                         "with the evidence when present).",
         source_url="https://cellebrite.com/the-101/resources/notebook/"),

    Case("cellebrite_ctf_2025", "Cellebrite CTF 2025: Anne Lockwood, Dexter King, Kevin Mallory", 2025, "ctf",
         "Cellebrite's 2025 capture-the-flag: extractions of three persons' phones.",
         "Cellebrite (via NIST CFReDS)",
         evidence=tuple(
             RE(f"cb2025_{cid}", f"{who}'s phone extraction", "mixed", (url,), password="B3tty&J0di_Swe@tta_W3aTHa",
                exclude=r"\.clbe$", transfer_gb=gb, group="extended")
             for cid, who, url, gb in (
                 ("anne", "Anne Lockwood", "https://drive.google.com/file/d/1Q9HZFYiB4iZx98DkNSv6MCTIOHfra8fB/view", 23.8),
                 ("kevin", "Kevin Mallory", "https://drive.google.com/file/d/1Qrq8q1xS4kxZBs4qGxlk90rtHHhn1ogj/view", 48.2),
                 ("dexter", "Dexter King", "https://drive.google.com/file/d/1jM5a7Qen1NKyY8ucOgNi0eJhnuj4qgxl/view", 74.7))),
         tags=("android", "ios", "multi-device", "ctf"),
         publisher_notes="Cellebrite CTF 2025: extractions of the phones of Anne Lockwood, Dexter King and Kevin "
                         "Mallory, with summary reports. The examiner answers the CTF questions about their "
                         "communications, movements and app activity.",
         source_url="https://cfreds.nist.gov (Cellebrite 2025 CTF)"),

    # ======================================================================== other published cases
    Case("nfi_rodeo_2023", "NFI Forensic Rodeo 2023 (DFRWS EU): Android device", 2023, "ctf",
         "For DFRWS EU 2023 the Netherlands Forensic Institute prepared a CTF: answer questions on data acquired from "
         "a specially prepared Android device.",
         "Netherlands Forensic Institute (via NIST CFReDS)",
         evidence=(RE("nfi2023_android", "Prepared Android device acquisition", "android",
                      ("https://drive.google.com/file/d/1srywys9Kn9MGtUeyNGAum3rVC8Bk_DLN/view",
                       "https://drive.google.com/file/d/1uz5x3wwBJJXMHdQZZR9TojGBLMdAiqd7/view"),
                      password="m5L5T!LSH7*8dN", extra_passwords=("kt6rYB!SfAU2#E",), transfer_gb=0.9),),
         tags=("android", "ctf", "single-device"),
         publisher_notes="For the DFRWS Europe 2023, the NFI prepared the digital forensic capture-the-flag (CTF) "
                         "called 2023 NFI Forensic Rodeo. The challenge is to answer questions on data acquired from an "
                         "Android device that was prepared for this event. (The readme is saved with the case.)",
         source_url="https://cfreds.nist.gov (2023 NFI Forensic Rodeo)"),

    Case("techhive", "The TechHive Scenario: stolen source code", 2026, "scenario",
         "Chad Turner, a contractor for Friendly Neighbor IT, stole source code and used it to start TechHive with "
         "his friend Stanley. Chad's Dell Inspiron 14 (Windows 11 on Arm) was imaged.",
         "Mark Spencer (via NIST CFReDS)",
         evidence=(RE("techhive_chad_laptop", "Chad Turner's Dell Inspiron 14 (Windows 11 on Arm), E01", "windows",
                      ("https://drive.google.com/file/d/195cj9Rlct60Pa60t6fVzjdqFyBVWSMM5/view",), transfer_gb=88.7,
                      group="extended"),),
         tags=("windows", "ip-theft", "insider", "single-device"),
         publisher_notes="The TechHive Scenario involves a Dell Inspiron 14 laptop running Windows 11 on Arm used by "
                         "Chad Turner, a contractor for Friendly Neighbor IT. Chad stole source code from his company and "
                         "used it to start a new business called TechHive with his friend Stanley. Your challenge is to "
                         "unravel Chad and Stanley's activities by analyzing the disk image obtained from Chad's laptop.",
         source_url="https://cfreds.nist.gov (The TechHive Scenario)"),

    Case("bart_the_hacker", "Bart the hacker (ForensicVM showcase, Windows 11)", 2023, "challenge",
         "A Windows 11 Pro system with local and cloud apps belonging to 'Bart': determine the nature of his password, "
         "his motives, the cloud and financial applications involved, and cryptocurrency-related information.",
         "Nuno Mourinho, Mario Candeias, Rogerio Bravo (ESTIG Beja), Zenodo",
         evidence=(RE("bart_win11", "Bart's Windows 11 Pro system (E01, 10 segments)", "windows", ("zenodo:10070981",),
                      transfer_gb=15.0, group="heavy"),),
         tags=("windows", "cloud", "cryptocurrency", "single-device"),
         publisher_notes="A Windows 11 Pro dataset designed in VirtualBox, complete with local and cloud apps, is to be "
                         "analyzed for crucial evidence. The task involves determining the nature of Bart's password, "
                         "Bart's motives, identifying involved cloud applications, and extracting data both offline and "
                         "online. The viability of dead box forensics for complete data retrieval is questioned, "
                         "alongside what additional information network access could unveil. The challenge includes "
                         "identifying two financial applications, extracting their data, and gathering "
                         "cryptocurrency-related information.",
         source_url="https://zenodo.org/doi/10.5281/zenodo.10070981"),

    Case("ou_red_petya_server", "Compromised Windows Server 2022 ('Red Petya' ransomware)", 2024, "scenario",
         "A simulated UK small-office Windows Server 2022 opened to RDP for home working was found on 12 Feb 2024 "
         "displaying 'Red Petya' ransomware; the disk was decrypted and imaged.",
         "Benjamin Donnachie, The Open University (figshare)",
         evidence=(RE("ou_winserver2022", "Decrypted Windows Server 2022 disk (E01)", "windows", ("figshare:26038642",),
                      transfer_gb=8.0, group="heavy"),),
         tags=("windows", "server", "ransomware", "rdp", "intrusion"),
         publisher_notes="Simulated UK-based small office network running from Sept 2023 to Feb 2024. The administrator "
                         "opened RDP to facilitate working from home. As part of the scenario, on 12th Feb 2024 it was "
                         "discovered the server was no longer responding with 'Red Petya' ransomware displayed on the "
                         "screen. Forensic experts were engaged, the disk decrypted and a forensic image taken in EnCase "
                         "E01 format. All information contained within the image (including usernames and IP "
                         "addresses) is synthetic.",
         source_url="https://doi.org/10.21954/ou.rd.26038642.v1"),

    Case("ou_defaced_webserver", "Defaced e-commerce web server (Ubuntu 22.04)", 2024, "scenario",
         "A simulated UK e-commerce site (DVWA used to simulate coding errors from 28 May 2024) was found defaced on "
         "the morning of 6 June 2024 and imaged.",
         "Benjamin Donnachie, The Open University (figshare)",
         evidence=(RE("ou_defaced_ubuntu", "Ubuntu 22.04 web server disk (E01)", "linux", ("figshare:26038669",),
                      transfer_gb=6.0, group="heavy"),),
         tags=("linux", "web-server", "defacement", "intrusion"),
         publisher_notes="Simulated UK-based ecommerce website. As part of the scenario, DVWA (Damned vulnerable web "
                         "app) was used to simulate coding errors on 28th May 2024. Morning of 6th June 2024 the system "
                         "was discovered to be defaced. Forensic experts were engaged and a forensic image taken in "
                         "EnCase E01 format. All information contained within the image (including usernames, credit "
                         "card numbers and IP addresses) is synthetic.",
         source_url="https://doi.org/10.21954/ou.rd.26038669.v1"),

    Case("cado_aws_eks", "Compromised AWS EKS Kubernetes cluster (Cado Security)", 2021, "scenario",
         "An AWS Kubernetes (EKS) cluster was compromised: a raw image of an EKS node plus Kubernetes API audit logs, "
         "VPC flow logs, CloudTrail and authenticator logs.",
         "Chris Doman / Cado Security (SANS DFIR 2021)",
         evidence=(RE("cado_eks_node", "EKS node raw image + AWS / Kubernetes logs", "linux",
                      ("https://github.com/cado-security/AWS_EKS_Cluster_Forensics/releases/download/v1.0/"
                       "CadoSecurity-CloudForensics-SANS_DFIR2021-Files.7z",), transfer_gb=1.2, group="heavy"),),
         tags=("cloud", "aws", "kubernetes", "linux", "intrusion"),
         publisher_notes="This investigation is based on an AWS Kubernetes cluster that has been compromised, and the "
                         "associated log sources that can be made available through AWS Cloud. The data sources include "
                         "the raw forensic image of the AWS EKS node (collected by Cado Response, gzipped dd), "
                         "kube-apiserver audit logs, AWS VPC flow logs for the cluster, CloudTrail logs and AWS "
                         "authenticator logs for the EKS role/cluster.",
         source_url="https://github.com/cado-security/AWS_EKS_Cluster_Forensics"),

    Case("dstl_ground_truth", "DSTL Ground Truth datasets: recovering data (exFAT, ext4, NTFS)", 2022, "reference_device",
         "Three small disk images (exFAT, ext4, NTFS) populated with documented content, imaged live and again after "
         "reformatting, with a ground-truth document of what should be recoverable.",
         "Holly Duns / DSTL (via NIST CFReDS)",
         evidence=(RE("dstl_images", "exFAT / ext4 / NTFS images, live and reformatted (E01)", "mixed",
                      ("https://drive.google.com/drive/folders/169IjXRyR_nKg3aZPAXLhiTgcJ3IZlzzY",),
                      include=r"\.E01$|Documentation/", answer_key=r"Ground Truth\.v|Ground_Truth|Annex",
                      transfer_gb=0.2),),
         tags=("recovery", "exfat", "ext4", "ntfs", "reformat", "ground-truth"),
         publisher_notes="Ground Truth Datasets - Recovering Data (DSTL): exFAT, ext4 and NTFS volumes populated with "
                         "known content and imaged while live, then reformatted and imaged again. The overview and "
                         "technical documents describe the population; the ground-truth document (answer key) lists "
                         "what each image contains.",
         source_url="https://cfreds.nist.gov (Ground Truth Datasets)"),

    Case("utfpr_linux_evidence", "UTFPR Linux forensics scenario: looking for evidences", 2023, "scenario",
         "A Linux machine in a fictional story of illegal images (stand-in photos from a public machine-learning "
         "dataset); several challenges to study and learn computer forensics.",
         "UTFPR (via NIST CFReDS)",
         evidence=(RE("utfpr_iso_dd", "Linux machine disk image (dd)", "linux",
                      ("https://drive.google.com/drive/folders/1_C-YorlEjuiCF6dBPmKhLd2Z7l43Q9YN",), include=r"^iso\.dd$",
                      transfer_gb=19.3, group="heavy"),),
         tags=("linux", "illegal-images", "single-device"),
         publisher_notes="It's a machine which contains several challenges in order for people to study, develop and "
                         "learn computer forensics; for the purpose of this scenario/challenge a fictional story was "
                         "created, and the images that will be considered illegal are just pistachio photos taken from "
                         "a machine learning dataset. The scenario can also be solved as a CTF on TryHackMe "
                         "(room 'lookingforevidencesrr').",
         source_url="https://tryhackme.com/room/lookingforevidencesrr"),

    Case("utfpr_server_hacked", "UTFPR Linux forensics scenario: a company's server was hacked", 2023, "scenario",
         "A simulated attack on a company's Linux server, built as a forensic study scenario with a fictional story.",
         "Jean Miguel / UTFPR (via NIST CFReDS)",
         evidence=(RE("utfpr_server_img", "Company server disk image", "linux",
                      ("https://drive.google.com/drive/folders/1eHqHqm8pwRXGC6cRZiI0Ls8YMs_qQP6t",), transfer_gb=18.3,
                      group="heavy"),),
         tags=("linux", "server", "intrusion", "single-device"),
         publisher_notes="It's a machine which contains some challenges in order for people to study and learn "
                         "computer forensics; for the purpose of this scenario/challenge a fictional story was created: "
                         "a company's server was hacked. Investigate how the attacker got in and what was done. The "
                         "scenario can also be solved as a CTF on TryHackMe (room 'acompanysserverwashacked').",
         source_url="https://tryhackme.com/room/acompanysserverwashacked"),

    *_DRONE_CASES,
)
