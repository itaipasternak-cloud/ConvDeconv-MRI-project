#!/bin/bash
# Full vs readout-cropped k-space fitting -- see readout_crop_test.py. Default: 8 tuning images x
# 3 TV weights x 2 modes = 48 Run A-regime fits; roughly 2-3 hours on one GPU.
#
# Submit:   sbatch run_readout_crop_test.sbatch        (extra args go to the script, e.g. --n-images 4)
# Log:      slurm_logs/crop-<jobid>.out (summary at the end)
# Results:  ~/fastmri_results/knee/diagnostics/readout_crop_test/results.csv
# Preemption-safe: every fit is appended to results.csv; a requeued job skips what's done.

#SBATCH --job-name=convdecoder_crop_test
#SBATCH --output=slurm_logs/crop-%j.out
#SBATCH --error=slurm_logs/crop-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --account=acct-preempt
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

mkdir -p slurm_logs
source ~/.bashrc
source /home/itai.p/miniforge3/etc/profile.d/conda.sh
conda activate base

export CD_ANATOMY=knee
export CONVDECODER_DATA_DIR=~/fastmri_data/knee_multicoil_combined
export CONVDECODER_CKPT_ROOT=~/fastmri_results/knee
export CD_K_AVERAGING=False
export CD_RUN_BATCH_EVAL=False
export CD_RUN_GRID_SEARCH=False
export CD_EARLY_STOP_PLOT=False
export CD_EARLY_STOP_PLOT_LIVE=False
export TQDM_MININTERVAL=60

echo "Job started on $(hostname) at $(date)"
python3 readout_crop_test.py "$@"
echo "Finished at $(date)."
