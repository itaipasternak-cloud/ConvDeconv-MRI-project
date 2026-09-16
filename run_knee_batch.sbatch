#!/bin/bash

# --- SLURM Settings ---
#SBATCH --job-name=convdecoder_knee_batch100
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.out
#SBATCH --account=acct-preempt        # <-- fill in your actual account/partition/qos
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00               # <-- 100 images x (K-ensemble fits + accel fits) is a lot
                                       #     more than a single-image run; adjust once you've
                                       #     timed a handful of images and can extrapolate.

# --- Setup Environment ---
mkdir -p slurm_logs outputs
source ~/.bashrc
source /path/to/your/miniforge3/etc/profile.d/conda.sh
conda activate your-env-name

# --- Dataset for THIS run: knee ---
export CD_ANATOMY=knee                 # feeds CKPT_ROOT + Section 6's data-folder resolution
# Matches the nested-subfolder pattern confirmed for brain (the archive extracts into a
# multicoil_val/ subfolder, not flat into knee_multicoil_val/ itself) -- verify once the knee
# download actually finishes: `find ~/fastmri_data/knee_multicoil_val -maxdepth 2 -type d`.
# CONVDECODER_DATA_DIR_KNEE (anatomy-specific) takes priority over CONVDECODER_DATA_DIR if both
# are set -- either name works for a knee-only run like this one.
export CONVDECODER_DATA_DIR=~/fastmri_data/knee_multicoil_val/multicoil_val

# Not required, but kept explicit: without CONVDECODER_CKPT_ROOT set, CKPT_ROOT now defaults to
# a per-ANATOMY subfolder automatically (Section 2), so knee/brain results can no longer collide
# even without this line -- it's just belt-and-braces.
export CONVDECODER_CKPT_ROOT=~/fastmri_results/knee

# --- Config overrides for this run ---
export CD_EARLY_STOP_PLOT_LIVE=False   # headless run -- see Section 2's note on this toggle
export CD_KAVG_BATCH_SIZE=100          # ~100 images, this run
# EVAL_TARGET_FILENAME defaults to a reproducible random pick from CD_ANATOMY's folder (Section
# 6/12.1) -- no need to pin one. Uncomment only if you want to force a specific, known file:
#   export CD_EVAL_TARGET_FILENAME=file1000041.h5
#   export CD_ARCHITECTURE=convdecoder_dcse
#   export CD_K_VALUE=5
#   export CD_RUN_MODE=new_fit
#   export CD_SEED=0

# --- Run Code ---
echo "Job started on $(hostname) at $(date)"

NOTEBOOK="MRI_ConvDeconv_variance_earlystop.ipynb"
OUT_DIR="outputs"
STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
FILTERED_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_${STAMP}_filtered.ipynb"
OUT_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_knee_${STAMP}.ipynb"

# Strips cells tagged "skip-slurm" -- this now includes Sections 7-11 (the single-image demo
# fits + their reporting cells), which are redundant with Section 12's own self-sufficient
# K-ensemble batch pipeline and would otherwise burn ~1-2 hours per job on a fit this run doesn't
# need, on top of Section 15's git-push cell (unattended-job-unsafe regardless of dataset).
jupyter nbconvert \
    --to notebook \
    --TagRemovePreprocessor.remove_cell_tags='["skip-slurm"]' \
    --output "$(basename "$FILTERED_NOTEBOOK")" \
    --output-dir "$OUT_DIR" \
    "$NOTEBOOK"

papermill \
    "$FILTERED_NOTEBOOK" \
    "$OUT_NOTEBOOK" \
    --log-output \
    --execution-timeout -1

rm -f "$FILTERED_NOTEBOOK"

echo "Finished at $(date)."
echo "Output notebook: $OUT_NOTEBOOK"
echo "Results CSV under: $CONVDECODER_CKPT_ROOT/results/"
