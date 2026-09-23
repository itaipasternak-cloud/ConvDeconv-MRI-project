#!/bin/bash

# 1 of 5 -- re-split of the 5x5=25-combination ACCEL_HUBER_DELTA/ACCEL_TV_WEIGHT grid search
# (superseding the earlier 2-way part1/part2 split, which put part1 at ~45.2h -- over a 30h cap).
# Splitting one delta value per job (5 tv_weight values each = 5 combinations/job) keeps every
# job comfortably under 30h: 5 combos x 13 images x 6000 iters x ~0.139 sec/iter =~ 15.1 hours.
# Same regime as before -- un-guided (random-init), ACCEL_NUM_ITERS=6000 (Run A's config), NOT
# the K=4 ensemble's guided/1350-iter regime.
#
# This job: huber_delta = 0.04 (all 5 tv_weight values).
#
# All 5 g1-g5 scripts are identical except GRID_HUBER_DELTA_VALUES/GRID_SEARCH_JOB_TAG/--job-name/
# output-notebook-suffix. Submit all 5 to run in parallel:
#   sbatch run_knee_grid_search_g1.sbatch
#   sbatch run_knee_grid_search_g2.sbatch
#   sbatch run_knee_grid_search_g3.sbatch
#   sbatch run_knee_grid_search_g4.sbatch
#   sbatch run_knee_grid_search_g5.sbatch
#
# Each job writes its own tagged output files (optuna_best_huber_tv_accel_*_g1.json ... _g5.json,
# and matching trial-history CSVs) rather than literally sharing one file -- five processes
# concurrently overwriting the SAME path would silently lose whichever finished writing last, so
# that isn't actually safe to do live. Once all 5 finish, run combine_grid_search_results.py
# (alongside this script) to merge all 5 into ONE combined trial-history CSV and ONE combined
# best-params JSON -- that's the "same output file" as a safe END STATE instead of a live race.
#
# N_TUNING_IMAGES stays at 13 (same as every prior version) so all 5 jobs read the existing
# 21-image tuning_filenames.json read-only, with no backfill-write race between them.

# --- SLURM Settings ---
#SBATCH --job-name=convdecoder_knee_grid_search_g1
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.out
#SBATCH --open-mode=append             # part-preempt kills + requeues the SAME job ID on
                                        # preemption -- without this, the requeued run would
                                        # truncate/overwrite this log instead of appending to it.
#SBATCH --requeue                      # part-preempt preemption kills the job outright (not
                                        # suspend/resume) -- this puts it back in the queue instead
                                        # of losing it. The grid search itself isn't checkpointed
                                        # mid-run, so a requeue re-runs this job's 5 combinations
                                        # from scratch -- but that's only ~15h lost, not ~75h.
#SBATCH --account=acct-preempt
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00               # <-- ~15.1h estimate + margin, comfortably under 30h.
                                       #     Still an ESTIMATE (job 92838's calibration, not
                                       #     measured at 6000 iterations specifically) -- watch the
                                       #     first trial's printed timing once running.

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

# --- Grid-search config (this job's slice: huber_delta=0.04 only) ---
export CD_RUN_ACCEL_GRID_SEARCH=True
export CD_N_TUNING_IMAGES=13
export CD_TUNING_PROBE_ITERS=6000
export CD_GRID_HUBER_DELTA_VALUES="[0.04]"
export CD_GRID_TV_WEIGHT_VALUES="[4e-06, 7e-06, 1.177e-05, 2e-05, 3.5e-05]"
export CD_GRID_SEARCH_JOB_TAG=g1

# --- Run Code ---
echo "Job started on $(hostname) at $(date)"
echo "GRID SEARCH 1/5: ACCEL_HUBER_DELTA/ACCEL_TV_WEIGHT, un-guided/6000-iter regime, delta=0.04 x all 5 tv values (5 combinations)."

NOTEBOOK="MRI_ConvDeconv_variance_earlystop.ipynb"
OUT_DIR="outputs"
STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
FILTERED_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_${STAMP}_filtered.ipynb"
OUT_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_knee_gridsearch_g1_${STAMP}.ipynb"

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
echo "Winning params (this slice): \$CONVDECODER_CKPT_ROOT/optuna_best_huber_tv_accel_*_g1.json"
echo "Trial history (this slice): \$CONVDECODER_CKPT_ROOT/tuning/optuna_trials_huber_tv_accel_*_g1.csv"
echo "Run combine_grid_search_results.py once all 5 g1-g5 jobs finish to merge into one file."
