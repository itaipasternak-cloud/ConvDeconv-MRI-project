#!/usr/bin/env python3
"""How many images of each (acquisition, coil count) group are still free, after everything the
pipeline has already reserved? Run on the cluster login node before a grid search or batch run.

Each experiment on one group needs, from that group's files:
    5 demo + 2 default target/reference images   (Section 6)
    K_VALUE reference images                      (Section 12, K=4)
    the fixed evaluation batch                    (Section 12.5, 100 images)
    the grid-search tuning images                 (Section 6.5, 30 images)
  = ~141 images, plus a few unreadable/failed ones.

Usage:
    python3 check_image_budget.py --data ~/fastmri_data/knee_multicoil_combined --results ~/fastmri_results/knee
    python3 check_image_budget.py --data ~/fastmri_data/brain_multicoil_val/multicoil_val --results ~/fastmri_results/brain
"""
import argparse
import glob
import json
import os
from collections import Counter

import h5py

NEEDED = 5 + 2 + 4 + 100 + 30


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="folder with the .h5 files")
    parser.add_argument("--results", help="CONVDECODER_CKPT_ROOT for this anatomy (for reserved images)")
    args = parser.parse_args()

    folder = os.path.expanduser(args.data)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".h5"))
    group_of, unreadable = {}, []
    for fname in files:
        try:
            with h5py.File(os.path.join(folder, fname), "r") as f:
                group_of[fname] = (f.attrs.get("acquisition"), f["kspace"].shape[1])
        except (OSError, KeyError):
            unreadable.append(fname)

    reserved = set()
    if args.results:
        sel = os.path.join(os.path.expanduser(args.results), "selections")
        for pattern in ("kavg_batch_selection_*.json", "kavg_ref_selection_*.json", "tuning_filenames.json"):
            for p in glob.glob(os.path.join(sel, pattern)):
                with open(p) as f:
                    data = json.load(f)
                reserved.update(data["filenames"] if isinstance(data, dict) else data)
        bad = os.path.join(sel, "kavg_batch_bad_files.json")
        if os.path.exists(bad):
            with open(bad) as f:
                reserved.update(json.load(f))

    total = Counter(group_of.values())
    free = Counter(g for f, g in group_of.items() if f not in reserved)
    print(f"{len(files)} .h5 files in {folder} ({len(unreadable)} unreadable)")
    print(f"{len(reserved)} images already reserved (batch, references, tuning, known-bad)\n")
    print(f"{'acquisition':<14}{'coils':>6}{'total':>8}{'free':>7}   enough for a new ~{NEEDED}-image experiment?")
    for (acq, coils), n in sorted(total.items(), key=lambda kv: -kv[1]):
        if n < 20:
            continue
        verdict = "yes" if n >= NEEDED + 5 else "no -- download more of this group"
        print(f"{str(acq):<14}{coils:>6}{n:>8}{free[(acq, coils)]:>7}   {verdict}")
    print("\n'free' = not in any saved selection (it still includes the ~7 demo/default images the "
          "notebook picks itself). A grid search on a group that already has its 100-image batch "
          "needs 30 of them.")


if __name__ == "__main__":
    main()
