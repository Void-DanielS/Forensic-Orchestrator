"""Pre-registered planner protocol: the user-level file definition and the "clearly beats LR" decision."""
import numpy as np

from src.ml.planner_protocol import is_container, paired_bootstrap, verdict


def test_container_definition():
    yes = ["fs/vol1/$MFT", "fs/$LogFile", "fs/$FAT1", "fs/pagefile.sys", "fs/ram.raw/ram.raw", "fs/x.pcap",
           "fs/Unallocated/0001.bin", "fs/mtd4_cache.bin", "fs/disk.E01"]
    no = ["fs/$Recycle.Bin/S-1-5/$IAB12.jpg", "fs/$Recycle.Bin/S-1-5/$RAB12.jpg", "fs/Users/a/Free_space_x.pdf",
          "fs/Users/a/NTUSER.DAT", "fs/Users/a/History", "fs/data/app.bin"]
    assert all(is_container({"path": p, "category": "other"}) for p in yes)
    assert not any(is_container({"path": p, "category": "user_document"}) for p in no)
    assert is_container({"path": "fs/a.log", "category": "network_capture"})


def test_verdict():
    lr = {f"c{i}": 0.3 for i in range(13)}
    good = {c: 0.4 for c in lr}
    v = verdict(good, lr)
    assert v["clearly_beats_lr"] and v["wins"] == 13 and v["ci95"][0] > 0
    mixed = {c: (1.0 if i == 0 else 0.299) for i, c in enumerate(lr)}       # one huge win, 12 small losses
    v = verdict(mixed, lr)
    assert v["mean_gain"] >= 0.05 and not v["clearly_beats_lr"] and v["wins"] == 1
    lo, hi = paired_bootstrap(np.zeros(13))
    assert lo == hi == 0.0
