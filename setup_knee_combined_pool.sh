#!/bin/bash
# Run this ONCE on the cluster (login node is fine -- it's just symlinking, no compute) after
# both knee_multicoil_val and knee_multicoil_train_batch_0 have finished downloading+extracting.
#
# Why: knee_multicoil_val alone has only 100 CORPD_FBK + 98 CORPDFS_FBK eligible files (see
# check_knee_pool.py's output) -- one short of a clean 100-image batch (run_knee_batch.sbatch's
# CD_KAVG_BATCH_SIZE=100) if the run happens to land on a CORPDFS_FBK target. This merges
# knee_multicoil_train_batch_0's files in via symlinks (no duplicate copies on disk) into a new
# knee_multicoil_combined/ folder, which run_knee_batch.sbatch now points CONVDECODER_DATA_DIR at.
#
# Safe to re-run: it wipes and rebuilds knee_multicoil_combined/ each time rather than appending,
# so stale symlinks from a previous run never linger.

set -euo pipefail

VAL_DIR=~/fastmri_data/knee_multicoil_val/multicoil_val
TRAIN_DIR=~/fastmri_data/knee_multicoil_train
COMBINED_DIR=~/fastmri_data/knee_multicoil_combined

for d in "$VAL_DIR" "$TRAIN_DIR"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: $d does not exist -- has it finished downloading/extracting?" >&2
        exit 1
    fi
done

rm -rf "$COMBINED_DIR"
mkdir -p "$COMBINED_DIR"

echo "Linking val files from $VAL_DIR ..."
find "$VAL_DIR" -maxdepth 1 -name '*.h5' -exec ln -s {} "$COMBINED_DIR"/ \;

echo "Linking train_batch_0 files from $TRAIN_DIR (searching subfolders) ..."
find "$TRAIN_DIR" -name '*.h5' -exec ln -s {} "$COMBINED_DIR"/ \;

N_LINKED=$(find "$COMBINED_DIR" -name '*.h5' | wc -l)
N_UNIQUE=$(find "$COMBINED_DIR" -name '*.h5' -printf '%f\n' | sort -u | wc -l)

echo
echo "Linked $N_LINKED files ($N_UNIQUE unique filenames) into $COMBINED_DIR"
if [ "$N_LINKED" -ne "$N_UNIQUE" ]; then
    echo "WARNING: filename collisions between val/train_batch_0 -- $((N_LINKED - N_UNIQUE)) duplicate name(s)." >&2
    echo "         (unexpected for fastMRI's disjoint file IDs -- investigate before trusting the combined pool)" >&2
fi

echo
echo "Next: re-run check_knee_pool.py pointed at $COMBINED_DIR to confirm both acquisition"
echo "groups now clear 100 eligible files."
