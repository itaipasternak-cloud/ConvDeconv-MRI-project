#!/bin/bash

# Runs Section 6.5's new exhaustive grid search over ACCEL_HUBER_DELTA/ACCEL_TV_WEIGHT
# (RUN_ACCEL_GRID_SEARCH, added alongside GRID_HUBER_DELTA_VALUES/GRID_TV_WEIGHT_VALUES in
# Section 2) and stops there -- K_AVERAGING/RUN_BATCH_EVAL are forced off below so Section 12's
# reference-checkpoint building and 100-image batch pipeline don't also run in this job.
#
# RUN_ACCEL_GRID_SEARCH itself, N_TUNING_IMAGES (21), and TUNING_PROBE_ITERS (1350) are NOT
# CD_*-overridable (see Section 2's env-override list -- only a subset of toggles are wired
# through it, and the grid-search toggles aren't among them), so they're already set directly in
# the notebook itself for this run: RUN_ACCEL_GRID_SEARCH=True, N_TUNING_IMAGES=21,
# TUNING_PROBE_ITERS=1350. Together with the 5x5=25-combination default grid
# (GRID_HUBER_DELTA_VALUES/GRID_TV_WEIGHT_VALUES), that's 25 x 21 x 1350 iterations x ~0.139
# sec/iteration (calibrated from job 92838's K=4 benchmark, single-fit here so no x4) =~ 27.4
# hours, leaving ~2.5h of this job's 30h budget as margin for the confirmation pass + startup
# overhead. Edit those three directly in the notebook (Section 2 / just above
# run_hyperparameter_search in Section 6.5) before submitting if you want a different grid
# size/image count/iteration budget.

# --- SLURM Settings ---
#SBATCH --job-name=convdecoder_knee_grid_search
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.out
#SBATCH --open-mode=append             # part-preempt kills + requeues the SAME job ID on
                                        # preemption -- without this, the requeued run would
                                        # truncate/overwrite this log instead of appending to it.
#SBATCH --requeue                      # part-preempt preemption kills the job outright (not
                                        # suspend/resume) -- this asks SLURM to put it back in
                                        # the queue afterward instead of just losing it. The grid
                                        # search itself isn't checkpointed mid-run (each Optuna
                                        # trial either completes or doesn't), so a requeue re-runs
                                        # this job from scratch -- still better than losing a ~30h
                                        # job outright to preemption.
#SBATCH --account=acct-preempt
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00               # <-- matches the 30h compute budget the grid was sized
                                       #     for (see math above); this is an ESTIMATE based on a
                                       #     single-K=4 benchmark scaled down to single-fit cost,
                                       #     not directly measured for this search -- watch the
                                       #     first few trials' timing (printed live per trial) and
                                       #     adjust N_TUNING_IMAGES down for a re-run if this looks
                                       #     like it'll miss the limit.

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
export CD_EARLY_STOP_PLOT=False        # same reasoning as the other batch scripts -- 25 grid
                                        # points x 21 images each is far too many plots to be useful

# Stop Section 12 (reference-checkpoint building + the 100-image batch pipeline) from also
# running after Section 6.5's grid search finishes -- this job is the search only. Re-run the
# existing run_knee_batch.sbatch separately afterward, under the winning ACCEL_HUBER_DELTA/
# ACCEL_TV_WEIGHT, to confirm the grid's pick with a real K-ensemble batch result.
export CD_K_AVERAGING=False
export CD_RUN_BATCH_EVAL=False

# --- Run Code ---
echo "Job started on $(hostname) at $(date)"
echo "Running Section 6.5's ACCEL_HUBER_DELTA/ACCEL_TV_WEIGHT grid search (RUN_ACCEL_GRID_SEARCH, N_TUNING_IMAGES=21, TUNING_PROBE_ITERS=1350, 25-combination default grid -- set directly in the notebook)."

NOTEBOOK="MRI_ConvDeconv_variance_earlystop.ipynb"
OUT_DIR="outputs"
STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
FILTERED_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_${STAMP}_filtered.ipynb"
OUT_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_knee_gridsearch_${STAMP}.ipynb"

# Strips "skip-slurm" (Sections 7-11's single-image demo -- these come AFTER Section 6.5 in the
# notebook and would otherwise burn extra time on a fit this job doesn't need) and
# "no-unattended" (Section 15's git-push cell), same as run_knee_batch.sbatch.
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
echo "Winning ACCEL_HUBER_DELTA/ACCEL_TV_WEIGHT printed near the end of that notebook's Section 6.5 output, and saved to \$CONVDECODER_CKPT_ROOT/optuna_best_huber_tv_accel_*.json"
echo "Full 25-trial history: \$CONVDECODER_CKPT_ROOT/tuning/optuna_trials_huber_tv_accel_*.csv"
