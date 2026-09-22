import h5py, os
from collections import Counter

folder = os.path.expanduser("~/fastmri_data/knee_multicoil_val/multicoil_val")
all_files = sorted(f for f in os.listdir(folder) if f.endswith(".h5"))

groups = Counter()
for fname in all_files:
    try:
        with h5py.File(os.path.join(folder, fname), 'r') as f:
            key = (f.attrs.get('acquisition'), f['kspace'].shape[1])
            groups[key] += 1
    except OSError as e:
        print(f"  SKIPPED {fname}: {e}")

print(f"Total .h5 files: {len(all_files)}")
for (acq, coils), count in sorted(groups.items(), key=lambda kv: -kv[1]):
    print(f"  acquisition={acq!r}, num_coils={coils}: {count} files")
