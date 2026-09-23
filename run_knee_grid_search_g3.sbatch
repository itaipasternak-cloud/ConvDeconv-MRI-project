#!/bin/bash

# 3 of 5 -- see run_knee_grid_search_g3.sbatch for the full explanation of the split, the
# CD_* env-var mechanism, and the shared-tuning-set caveat. This job: huber_delta = 0.1192 (all 5
# tv_weight values).

# --- SLURM Settings ---
#SBATCH --job-name=convdecoder_knee_grid_search_g3
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

# --- Grid-search config (this job's slice: huber_delta=0.1192 only) ---
export CD_RUN_ACCEL_GRID_SEARCH=True
export CD_N_TUNING_IMAGES=13
export CD_TUNING_PROBE_ITERS=6000
export CD_GRID_HUBER_DELTA_VALUES="[0.1192]"
export CD_GRID_TV_WEIGHT_VALUES="[4e-06, 7e-06, 1.177e-05, 2e-05, 3.5e-05]"
export CD_GRID_SEARCH_JOB_TAG=g3

# --- Run Code ---
echo "Job started on $(hostname) at $(date)"
echo "GRID SEARCH 3/5: ACCEL_HUBER_DELTA/ACCEL_TV_WEIGHT, un-guided/6000-iter regime, delta=0.1192 x all 5 tv values (5 combinations)."

NOTEBOOK="MRI_ConvDeconv_variance_earlystop.ipynb"
OUT_DIR="outputs"
STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
FILTERED_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_${STAMP}_filtered.ipynb"
OUT_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_knee_gridsearch_g3_${STAMP}.ipynb"

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
echo "Winning params (this slice): \$CONVDECODER_CKPT_ROOT/optuna_best_huber_tv_accel_*_g3.json"
echo "Trial history (this slice): \$CONVDECODER_CKPT_ROOT/tuning/optuna_trials_huber_tv_accel_*_g3.csv"
echo "Run combine_grid_search_results.py once all 5 g3-g5 jobs finish to merge into one file."
