#!/bin/bash

# --- SLURM Settings ---
#SBATCH --job-name=convdecoder_knee_batch
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.out
#SBATCH --open-mode=append             # part-preempt kills + requeues the SAME job ID on
                                        # preemption -- without this, the requeued run would
                                        # truncate/overwrite this log instead of appending to it.
#SBATCH --requeue                      # part-preempt preemption kills the job outright (not
                                        # suspend/resume) -- this asks SLURM to put it back in
                                        # the queue afterward instead of just losing it. Safe
                                        # here since the pipeline's own resume-by-CSV +
                                        # RUN_MODE=load_existing logic picks up where it left off.
#SBATCH --account=acct-preempt
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00               # <-- ~12.5 min/image once the K reference checkpoints are
                                       #     cached (job 92838) extrapolates to ~21h for 100
                                       #     images; bumped from 24h to 30h for margin, since
                                       #     per-image time can vary (matrix size, convergence).

# --- Setup Environment ---
mkdir -p slurm_logs outputs
source ~/.bashrc
source /home/itai.p/miniforge3/etc/profile.d/conda.sh
conda activate base

# --- Dataset for THIS run: knee ---
export CD_ANATOMY=knee                 # feeds CKPT_ROOT + Section 6's data-folder resolution
# knee_multicoil_val alone has only 100 CORPD_FBK + 98 CORPDFS_FBK eligible files -- one image
# short of covering a 100-image batch (CD_KAVG_BATCH_SIZE below) if the run happens to pick a
# CORPDFS_FBK target. knee_multicoil_combined/ merges in knee_multicoil_train_batch_0 (symlinks,
# built once via the commands in the PR/commit message -- see setup_knee_combined_pool.sh) to
# top up both groups. Re-run check_knee_pool.py against this folder after rebuilding the symlinks
# if the train batch ever changes.
# CONVDECODER_DATA_DIR_KNEE (anatomy-specific) takes priority over CONVDECODER_DATA_DIR if both
# are set -- either name works for a knee-only run like this one.
export CONVDECODER_DATA_DIR=~/fastmri_data/knee_multicoil_combined

# Not required, but kept explicit: without CONVDECODER_CKPT_ROOT set, CKPT_ROOT now defaults to
# a per-ANATOMY subfolder automatically (Section 2), so knee/brain results can no longer collide
# even without this line -- it's just belt-and-braces.
export CONVDECODER_CKPT_ROOT=~/fastmri_results/knee

# --- Config overrides for this run ---
export CD_EARLY_STOP_PLOT_LIVE=False   # headless run -- see Section 2's note on this toggle
# With up to 100 images x K-ensemble members each fit() call, even one final snapshot per fit
# would be far too many plots to be useful -- skip the early-stop diagnostic plot entirely for
# this batch run. Single-image sanity checks (run_single_image_test.sbatch) leave this on.
export CD_EARLY_STOP_PLOT=False

# Batch size defaults to 100 (the real run), but takes this script's first positional argument
# as an override -- no file editing needed to try a quick test first:
#   sbatch run_knee_batch.sbatch      # real run, 100 images
#   sbatch run_knee_batch.sbatch 1    # quick test, 1 image
# Results are keyed by RUN_TAG only (not batch size -- see Section 12.5's own comment on this),
# so a small test run's completed image(s) accumulate into the eventual full run's results CSV
# instead of being wasted or overwritten.
export CD_KAVG_BATCH_SIZE="${1:-100}"
# EVAL_TARGET_FILENAME defaults to a reproducible random pick from CD_ANATOMY's folder (Section
# 6/12.1) -- no need to pin one. Uncomment only if you want to force a specific, known file:
#   export CD_EVAL_TARGET_FILENAME=file1000041.h5
#   export CD_ARCHITECTURE=convdecoder_dcse
#   export CD_K_VALUE=5
#   export CD_RUN_MODE=new_fit
#   export CD_SEED=0

# --- Run Code ---
echo "Job started on $(hostname) at $(date)"
echo "KAVG_BATCH_SIZE for this run: $CD_KAVG_BATCH_SIZE"

NOTEBOOK="MRI_ConvDeconv_variance_earlystop.ipynb"
OUT_DIR="outputs"
STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
FILTERED_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_${STAMP}_filtered.ipynb"
OUT_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_knee_${STAMP}.ipynb"

# Strips cells tagged "skip-slurm" (Sections 7-11, the single-image demo fits + their reporting
# cells -- redundant with Section 12's own self-sufficient K-ensemble batch pipeline and would
# otherwise burn ~1-2 hours per job on a fit this run doesn't need) and "no-unattended" (Section
# 15's git-push cell -- unsafe in ANY unattended job, batch or single-image). See
# run_single_image_test.sbatch for the complementary script that keeps "skip-slurm" cells and
# strips only "no-unattended".
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
