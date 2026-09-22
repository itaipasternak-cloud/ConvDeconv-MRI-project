#!/bin/bash

# Run A baseline batch: a SINGLE plain random-init fit per image (no K-ensembling, no guided
# init), over the same 100-image knee target list as run_knee_batch.sbatch -- for a fair, paired
# comparison against the K-averaged ConvDecoder-A batch result. This reuses the K-averaging batch
# pipeline itself (Section 12.5) with K_VALUE=1 + KAVG_INIT_MODE="random" + ACCEL_NUM_ITERS
# raised to 6000 (per r-1.3), instead of the notebook's old, now-removed, separate
# Section 13 "Run A batch eval" code path -- see the notebook's own history for why that section
# was removed. Because RUN_TAG (architecture/Z_SOURCE/loss config) is unchanged from
# run_knee_batch.sbatch and KAVG_BATCH_KEEP_FIXED_SET defaults to True, this reuses the EXACT
# SAME 100 selected images as that run, rather than drawing a fresh random set.

# --- SLURM Settings ---
#SBATCH --job-name=convdecoder_knee_batch_runA
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
#SBATCH --time=30:00:00               # <-- ROUGH estimate, not yet measured directly: job 92838
                                       #     showed ~12.5 min/image for K=4 accelerated members
                                       #     at 1350 iter each (5400 total iterations/image).
                                       #     This run does 1 member at 6000 iterations/image --
                                       #     close to that same total iteration count, so
                                       #     ~14 min/image is a reasonable guess -> ~23h for 100
                                       #     images. Padded to 30h (same margin as
                                       #     run_knee_batch.sbatch); watch the first few images'
                                       #     timing and adjust if this run looks like it'll miss
                                       #     the limit.

# --- Setup Environment ---
mkdir -p slurm_logs outputs
source ~/.bashrc
source /home/itai.p/miniforge3/etc/profile.d/conda.sh
conda activate base

# --- Dataset for THIS run: knee (same as run_knee_batch.sbatch) ---
export CD_ANATOMY=knee
export CONVDECODER_DATA_DIR=~/fastmri_data/knee_multicoil_combined
export CONVDECODER_CKPT_ROOT=~/fastmri_results/knee

# --- Config overrides for this run ---
export CD_EARLY_STOP_PLOT_LIVE=False   # headless run -- see Section 2's note on this toggle
# Same reasoning as run_knee_batch.sbatch: 100 images is far too many plots to be useful.
export CD_EARLY_STOP_PLOT=False

# K_AVERAGING=True + RUN_BATCH_EVAL=True are REQUIRED even for this random-init-only run --
# Section 12.5.1's target-image selection (which this reuses) only executes when both are True,
# regardless of KAVG_INIT_MODE. KAVG_BUILD_REF_CHECKPOINTS is left at its default (True) but
# costs ~nothing here: pure random init needs zero reference checkpoints built (see Section 2's
# note), so that part is a fast no-op.
export CD_K_AVERAGING=True
export CD_RUN_BATCH_EVAL=True
export CD_K_VALUE=1                    # single member -- no ensembling/averaging
export CD_KAVG_INIT_MODE=random        # plain random init, not guided from a reference fit
export CD_ACCEL_NUM_ITERS=6000         # iteration count for this baseline (default accelerated
                                        # value is 1350 -- too short for a fair single-fit
                                        # baseline); 6000 per r-1.3

# Same positional-arg override pattern as run_knee_batch.sbatch, for a quick test before the
# real 100-image run:
#   sbatch run_knee_batch_runA.sbatch      # real run, 100 images
#   sbatch run_knee_batch_runA.sbatch 1    # quick test, 1 image
export CD_KAVG_BATCH_SIZE="${1:-100}"

# --- Run Code ---
echo "Job started on $(hostname) at $(date)"
echo "KAVG_BATCH_SIZE for this run: $CD_KAVG_BATCH_SIZE (Run A baseline: K=1, random init, ${CD_ACCEL_NUM_ITERS} iter)"

NOTEBOOK="MRI_ConvDeconv_variance_earlystop.ipynb"
OUT_DIR="outputs"
STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
FILTERED_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_${STAMP}_filtered.ipynb"
OUT_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_knee_runA_${STAMP}.ipynb"

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
