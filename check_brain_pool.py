"""
Eligibility-pool check for brain multicoil val data, analogous to check_knee_pool.py.

Section 12's K-ensemble batch eval needs, for each target image, K_VALUE other images that
match its `acquisition` attribute AND coil count (find_kavg_eligible_pool in the notebook --
guided-init reference checkpoints are only reusable across images with an identical k-space
schema). Knee has just 2 acquisition types (CORPD_FBK / CORPDFS_FBK); brain has several
(AXT1, AXT1POST, AXT1PRE, AXT2, AXFLAIR, ...) and likely more coil-count variation, so its
per-group counts could be much smaller than the 199-file total suggests -- this only tells you
by actually counting.

Run this BEFORE assuming brain_multicoil_val (batch_0/1/2, already merged into one folder per
run_brain_batch.sbatch's CONVDECODER_DATA_DIR) can support a CD_KAVG_BATCH_SIZE=100 run. If any
group you care about comes up short, that's the trigger to request/download more brain data and
build a brain_multicoil_combined/ the same way setup_knee_combined_pool.sh did for knee -- don't
do that pre-emptively, since it means more downloading, and it may not be necessary.
"""
import h5py
import os
from collections import Counter

folder = os.path.expanduser("~/fastmri_data/brain_multicoil_val/multicoil_val")
all_files = sorted(f for f in os.listdir(folder) if f.endswith(".h5"))

groups = Counter()
skipped = []
for fname in all_files:
    try:
        with h5py.File(os.path.join(folder, fname), 'r') as f:
            key = (f.attrs.get('acquisition'), f['kspace'].shape[1])
            groups[key] += 1
    except OSError as e:
        skipped.append((fname, str(e)))

print(f"Total .h5 files: {len(all_files)}")
for (acq, coils), count in sorted(groups.items(), key=lambda kv: -kv[1]):
    flag = "  <-- fewer than 100: won't cover a full 100-image batch on its own" if count < 100 else ""
    print(f"  acquisition={acq!r}, num_coils={coils}: {count} files{flag}")

if skipped:
    print(f"\n{len(skipped)} file(s) skipped (unreadable):")
    for fname, err in skipped:
        print(f"  SKIPPED {fname}: {err}")
