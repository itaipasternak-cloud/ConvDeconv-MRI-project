#!/bin/bash
# Hyperparameter grid search (notebook Section 6.5) for HUBER_DELTA/TV_WEIGHT, as ONE SLURM array.
#
# 5 x 5 = 25 (huber_delta, tv_weight) combinations x 30 tuning images = 750 fits of 6000
# iterations (Run A's regime: single random-init fit, scored exactly like the 100-image batch).
# The work is split into 12 shards, one per array task; at most 12 tasks run at once (%12), so
# this never takes more than 12 GPUs of the shared cluster.
#
# Estimated cost: ~14 min per fit (Run A batch estimate) -> ~175 GPU-hours total, ~15h per task.
# --time is 24h for margin; watch the first few "(...s, ETA ...h)" lines in the logs.
#
# Preemption-safe: every fit is appended to results_shard<i>.csv as soon as it finishes, and a
# requeued task skips what it already did -- a preemption costs at most the fit in progress.
#
# Submit:   sbatch run_knee_grid_search.sbatch
# Progress: tail slurm_logs/grid-<jobid>_<task>.out
# Results:  python3 analyze_grid_search.py --grid-dir ~/fastmri_results/knee/grid_search/grid_v2 --plot
#
# If you change the number of tasks, change BOTH --array and GRID_NUM_SHARDS below.
#
# Rerunning ONE shard that failed (e.g. shard 5), split over several GPUs so it finishes faster --
# fits it already recorded are skipped:
#   sbatch --array=0-3 --export=ALL,RERUN_SHARD=5,RERUN_PARTS=4 run_knee_grid_search.sbatch
# (--array must be 0 .. RERUN_PARTS-1.)

# --- SLURM Settings ---
#SBATCH --job-name=convdecoder_knee_grid
#SBATCH --array=0-11%12
#SBATCH --output=slurm_logs/grid-%A_%a.out
#SBATCH --error=slurm_logs/grid-%A_%a.out
#SBATCH --open-mode=append             # a requeued task appends to its log instead of truncating it
#SBATCH --requeue                      # part-preempt kills preempted jobs; put them back in the queue
#SBATCH --account=acct-preempt
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

GRID_NUM_SHARDS=12

# --- Setup Environment ---
mkdir -p slurm_logs outputs
source ~/.bashrc
source /home/itai.p/miniforge3/etc/profile.d/conda.sh
conda activate base

TASK="${SLURM_ARRAY_TASK_ID:-0}"
if [ -n "$RERUN_SHARD" ]; then
    # Rerun mode: this task is part $TASK of $RERUN_PARTS of shard $RERUN_SHARD.
    export CD_GRID_SHARD_INDEX="$RERUN_SHARD"
    export CD_GRID_SUB_INDEX="$TASK"
    export CD_GRID_SUB_COUNT="${RERUN_PARTS:?set RERUN_PARTS together with RERUN_SHARD}"
    LIMIT="$RERUN_PARTS"
else
    export CD_GRID_SHARD_INDEX="$TASK"
    LIMIT="$GRID_NUM_SHARDS"
fi
# Checks the task NUMBER, not the task count, so resubmitting a subset (e.g. --array=5) works.
if [ "$TASK" -ge "$LIMIT" ]; then
    echo "ERROR: array task $TASK is out of range (must be below $LIMIT) -- check --array." >&2
    exit 1
fi

# --- Dataset: knee (same as run_knee_batch.sbatch) ---
export CD_ANATOMY=knee
export CONVDECODER_DATA_DIR=~/fastmri_data/knee_multicoil_combined
export CONVDECODER_CKPT_ROOT=~/fastmri_results/knee

# --- Config overrides ---
export CD_EARLY_STOP_PLOT_LIVE=False
export CD_EARLY_STOP_PLOT=False
export CD_K_AVERAGING=False            # no Section 12 (reference fits / batch) in this job
export CD_RUN_BATCH_EVAL=False

# --- Grid search (grid values, image count and iterations come from Section 2's defaults) ---
export CD_RUN_GRID_SEARCH=True
export CD_GRID_SEARCH_NAME=grid_v2
export CD_GRID_NUM_SHARDS="$GRID_NUM_SHARDS"

# --- Run Code ---
echo "Job started on $(hostname) at $(date) -- grid shard $CD_GRID_SHARD_INDEX of $CD_GRID_NUM_SHARDS${RERUN_SHARD:+ (part $CD_GRID_SUB_INDEX of $CD_GRID_SUB_COUNT)}"

NOTEBOOK="MRI_ConvDeconv_variance_earlystop.ipynb"
OUT_DIR="outputs"
STAMP="${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d_%H%M%S)}_${CD_GRID_SHARD_INDEX}${RERUN_SHARD:+_part${CD_GRID_SUB_INDEX}}"
FILTERED_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_${STAMP}_filtered.ipynb"
OUT_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_knee_grid_${STAMP}.ipynb"

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
