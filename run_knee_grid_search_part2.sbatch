#!/bin/bash

# PART 2 of 2 -- runs alongside run_knee_grid_search_part1.sbatch. This job covers the other 2 of
# the 5 huber_delta values (all 5 tv_weight values each) = 10 combinations. See part1's script for
# the full explanation of the split, the CD_* env-var mechanism, and the shared-tuning-set caveat.
#
# 10 combos x 13 images x 6000 iters x ~0.139 sec/iter =~ 30.1 hours -- finishes well before part1
# (~45.2h), which is the long pole of this split.

# --- SLURM Settings ---
#SBATCH --job-name=convdecoder_knee_grid_search_part2
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --account=acct-preempt
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00               # <-- ~30.1h estimate + margin. CHECK YOUR PARTITION'S
                                       #     MaxTime FIRST (`scontrol show partition part-preempt`).
                                       #     Same estimate caveats as part1.

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
export CD_K_AVERAGING=False
export CD_RUN_BATCH_EVAL=False

# --- Grid-search config (this job's slice) ---
export CD_RUN_GRID_SEARCH=True
export CD_N_TUNING_IMAGES=13
export CD_TUNING_PROBE_ITERS=6000
export CD_GRID_HUBER_DELTA_VALUES="[0.2, 0.36]"
export CD_GRID_TV_WEIGHT_VALUES="[4e-06, 7e-06, 1.177e-05, 2e-05, 3.5e-05]"
export CD_GRID_SEARCH_JOB_TAG=part2

# --- Run Code ---
echo "Job started on $(hostname) at $(date)"
echo "PART 2/2: ACCEL_HUBER_DELTA/ACCEL_TV_WEIGHT grid search, un-guided/6000-iter regime, delta in [0.2, 0.36] x all 5 tv values (10 combinations)."

NOTEBOOK="MRI_ConvDeconv_variance_earlystop.ipynb"
OUT_DIR="outputs"
STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
FILTERED_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_${STAMP}_filtered.ipynb"
OUT_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_knee_gridsearch_part2_${STAMP}.ipynb"

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
echo "Winning params (this slice): \$CONVDECODER_CKPT_ROOT/optuna_best_huber_tv_accel_*_part2.json"
echo "Trial history (this slice): \$CONVDECODER_CKPT_ROOT/tuning/optuna_trials_huber_tv_accel_*_part2.csv"
echo "Combine with part1's results (same paths, _part1 suffix) once both jobs finish to get the true grid-wide winner."
