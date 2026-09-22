#!/bin/bash

# --- SLURM Settings ---
#SBATCH --job-name=convdecoder_brain_batch100
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.out
#SBATCH --account=acct-preempt
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00               # <-- adjust once you've timed a handful of brain images

# --- Setup Environment ---
mkdir -p slurm_logs outputs
source ~/.bashrc
source /home/itai.p/miniforge3/etc/profile.d/conda.sh
conda activate base

# --- Dataset for THIS run: brain ---
export CD_ANATOMY=brain                # feeds CKPT_ROOT + Section 6's data-folder resolution
# Confirmed from the actual completed download: both batch_0 and batch_1 extracted into the
# SAME nested multicoil_val/ subfolder under brain_multicoil_val/, so both batches' files are
# already visible from this one path -- no merging needed.
# CONVDECODER_DATA_DIR_BRAIN (anatomy-specific) takes priority over CONVDECODER_DATA_DIR if both
# are set -- either name works for a brain-only run like this one.
export CONVDECODER_DATA_DIR=~/fastmri_data/brain_multicoil_val/multicoil_val

# Not required, but kept explicit: without CONVDECODER_CKPT_ROOT set, CKPT_ROOT now defaults to
# a per-ANATOMY subfolder automatically (Section 2), so this line is really just belt-and-braces.
export CONVDECODER_CKPT_ROOT=~/fastmri_results/brain

# --- Config overrides for this run ---
export CD_EARLY_STOP_PLOT_LIVE=False   # headless run -- see Section 2's note on this toggle
# With up to 100 images x K-ensemble members each fit() call, even one final snapshot per fit
# would be far too many plots to be useful -- skip the early-stop diagnostic plot entirely for
# this batch run. Single-image sanity checks (run_single_image_test.sbatch) leave this on.
export CD_EARLY_STOP_PLOT=False
export CD_KAVG_BATCH_SIZE=100          # ~100 images, this run
# No longer required: EVAL_TARGET_FILENAME now defaults to a reproducible random pick from
# whatever CD_ANATOMY's folder contains (Section 6/12.1) -- brain filenames don't follow the
# same fileNNNNNNN.h5 pattern knee does, so hand-picking one was real friction. Uncomment this
# ONLY if you want to pin a specific, known file instead of the auto-picked default (e.g. to
# exactly reproduce a prior run):
#   export CD_EVAL_TARGET_FILENAME=file_brain_AXT1_...replace_with_a_real_filename....h5
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
OUT_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_brain_${STAMP}.ipynb"

# Strips cells tagged "skip-slurm" (Sections 7-11, the single-image demo fits + their reporting
# cells -- redundant with Section 12's own self-sufficient K-ensemble batch pipeline AND hardcode
# a knee-only filename with no fallback, so without this a brain run would crash partway through
# Section 7 before ever reaching batch eval) and "no-unattended" (Section 15's git-push cell --
# unsafe in ANY unattended job). See run_single_image_test.sbatch for the complementary script
# that keeps "skip-slurm" cells and strips only "no-unattended".
jupyter nbconvert \
    --to notebook \
    --TagRemovePreprocessor.remove_cell_tags='["skip-slurm", "no-unattended"]' \
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
