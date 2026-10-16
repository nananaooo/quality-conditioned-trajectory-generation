"""Download the four public LeRobot corpora at the exact revisions used in the paper (joint data only: data/ and meta/,
no videos) and build the (state, action) windows (length 60, stride 30, never crossing episodes).

    python scripts/prepare_data.py            # download + build windows for all four corpora
    python scripts/prepare_data.py --skip-download

Outputs
  data_raw/<corpus dir>/{data,meta}
  data_processed/windows/<name>_stateaction_w60_s30.npy
  data_processed/stats/<name>_stateaction_w60_s30_stats.npz   (includes window -> episode ids for episode-level splits)

The datasets are NOT redistributed here; please check each dataset's own terms on the Hugging Face Hub.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]

# name used in the code, Hub repo, pinned commit, sub-directory inside the repo (if any), local directory
CORPORA = [
    ("being_so101", "BeingBeyond/Being-H-EDU_SO101", "8bd76e337ab3b5abd482d8dfa9c04277b95e5bd9",
     "pick_cube_plate_v3.0", "data_raw/external/being_h_edu_so101/pick_cube_plate_v3_0"),
    ("rayenghali", "Rayenghali/combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30",
     "c1250a1f7f1a0e807aca12dd89119e04610d4a13", "", "data_raw/external/rayenghali_lego_cup"),
    ("koch_pickplace5", "lerobot/koch_pick_place_5_lego", "af1f4ff76be9fa752424b176125119744a6ef3f6", "",
     "data_raw/koch_pick_place_5_lego"),
    ("aloha_tape", "lerobot/aloha_static_tape", "92955878beb0da15eee5d4d869841577f87e903e", "",
     "data_raw/aloha_static_tape"),
]


def download(name, repo, rev, sub, out):
    out = ROOT / out
    prefix = f"{sub}/" if sub else ""
    snap = ROOT / "data_raw" / ".hf" / name          # short path: the Hub cache adds long file names (Windows MAX_PATH)
    snapshot_download(repo_id=repo, repo_type="dataset", revision=rev, local_dir=str(snap),
                      allow_patterns=[f"{prefix}data/**", f"{prefix}meta/**", f"{prefix}README.md", "README.md"])
    src = snap / sub if sub else snap
    for part in ("data", "meta"):
        if (out / part).exists():
            shutil.rmtree(out / part)
        shutil.copytree(src / part, out / part)
    for readme in (src / "README.md", snap / "README.md"):
        if readme.exists():
            shutil.copy2(readme, out / "README.md"); break


def build(name, root):
    cmd = [sys.executable, str(ROOT / "scripts" / "build_state_windows.py"), "--root", str(ROOT / root),
           "--window-size", "60", "--stride", "30",
           "--output", str(ROOT / f"data_processed/windows/{name}_stateaction_w60_s30.npy"),
           "--stats-output", str(ROOT / f"data_processed/stats/{name}_stateaction_w60_s30_stats.npz")]
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-download", action="store_true")
    a = p.parse_args()
    for name, repo, rev, sub, out in CORPORA:
        if not a.skip_download:
            print(f"[download] {repo}@{rev[:7]}", flush=True)
            download(name, repo, rev, sub, out)
        print(f"[windows] {name}", flush=True)
        build(name, out)


if __name__ == "__main__":
    main()
