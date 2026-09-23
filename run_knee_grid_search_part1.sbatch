#!/bin/bash

# PART 1 of 2 -- runs alongside run_knee_grid_search_part2.sbatch to split the 5x5=25-combination
# ACCEL_HUBER_DELTA/ACCEL_TV_WEIGHT grid search across two simultaneous jobs instead of one ~75h
# sequential run. Split along the delta axis: this job covers 3 of the 5 huber_delta values (all
# 5 tv_weight values each) = 15 combinations; part 2 covers the other 2 deltas = 10 combinations.
# Same regime as the single-job version -- un-guided (random-init), ACCEL_NUM_ITERS=6000 (Run A's
# config), NOT the K=4 ensemble's guided/1350-iter regime.
#
# All the grid-search toggles (RUN_ACCEL_GRID_SEARCH, N_TUNING_IMAGES, TUNING_PROBE_ITERS,
# GRID_HUBER_DELTA_VALUES, GRID_TV_WEIGHT_VALUES, GRID_SEARCH_JOB_TAG) are now CD_*-overridable
# (added alongside this split), so unlike the single-job script this one does NOT require editing
# the notebook directly -- everything's driven by the exports below, and both jobs can run off the
# exact same committed notebook.
#
# GRID_SEARCH_JOB_TAG=part1 makes the winning-params JSON and trial-history CSV land at distinct
# paths from part2's (optuna_best_huber_tv_accel_..._part1.json vs _part2.json, and the matching
# .csv) -- without this, both jobs would race to overwrite the SAME output files.
#
# N_TUNING_IMAGES is deliberately kept at 13 (same as part2, same as the original single-job run)
# rather than raised to use the freed-up-by-parallelism time -- this reuses the EXISTING
# tuning_filenames.json (already has 21 images, drawn by job 93088) read-only, with no need to
# backfill/rewrite it. If you raise N_TUNING_IMAGES above 21 for either part job, both jobs would
# race to backfill-and-rewrite that shared file -- submit sequentially (or delete + let one job
# redraw it, then submit the other) if you want to do that.
#
# 15 combos x 13 images x 6000 iters x ~0.139 sec/iter =~ 45.2 hours (vs part2's ~30.1h) -- these
# don't finish at the same time; part1 is the long pole. Total wall time across both parts is
# ~45.2h (this job), vs ~75.3h for the single-job version -- not quite the full 2x from perfectly
# balanced parallelism, since 25 doesn't split evenly across a 5-value axis.

# --- SLURM Settings ---
#SBATCH --job-name=convdecoder_knee_grid_search_part1
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
                                        # this job from scratch.
#SBATCH --account=acct-preempt
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=50:00:00               # <-- ~45.2h estimate + margin. CHECK YOUR PARTITION'S
                                       #     MaxTime FIRST (`scontrol show partition part-preempt`)
                                       #     -- may be rejected/capped lower than 50h. This is an
                                       #     ESTIMATE (job 92838's calibration, not measured at
                                       #     6000 iterations specifically) -- watch the first
                                       #     couple of trials' printed timing once running.

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
export CD_EARLY_STOP_PLOT_LIVE=False
export CD_EARLY_STOP_PLOT=False
export CD_K_AVERAGING=False            # stop Section 12 (ref-checkpoint building + 100-image
export CD_RUN_BATCH_EVAL=False         # batch pipeline) from also running after the search

# --- Grid-search config (this job's slice) ---
export CD_RUN_ACCEL_GRID_SEARCH=True
export CD_N_TUNING_IMAGES=13
export CD_TUNING_PROBE_ITERS=6000
export CD_GRID_HUBER_DELTA_VALUES="[0.04, 0.07, 0.1192]"
export CD_GRID_TV_WEIGHT_VALUES="[4e-06, 7e-06, 1.177e-05, 2e-05, 3.5e-05]"
export CD_GRID_SEARCH_JOB_TAG=part1

# --- Run Code ---
echo "Job started on $(hostname) at $(date)"
echo "PART 1/2: ACCEL_HUBER_DELTA/ACCEL_TV_WEIGHT grid search, un-guided/6000-iter regime, delta in [0.04, 0.07, 0.1192] x all 5 tv values (15 combinations)."

NOTEBOOK="MRI_ConvDeconv_variance_earlystop.ipynb"
OUT_DIR="outputs"
STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
FILTERED_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_${STAMP}_filtered.ipynb"
OUT_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_knee_gridsearch_part1_${STAMP}.ipynb"

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
echo "Winning params (this slice): \$CONVDECODER_CKPT_ROOT/optuna_best_huber_tv_accel_*_part1.json"
echo "Trial history (this slice): \$CONVDECODER_CKPT_ROOT/tuning/optuna_trials_huber_tv_accel_*_part1.csv"
echo "Combine with part2's results (same paths, _part2 suffix) once both jobs finish to get the true grid-wide winner."
