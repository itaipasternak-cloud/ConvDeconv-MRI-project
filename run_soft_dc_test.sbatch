#!/bin/bash
# Soft vs hard data consistency, re-scoring a finished batch run's saved fits (no fitting) -- see
# soft_dc_test.py. 100 images x 8 lambdas: roughly 30-60 minutes on one GPU.
# Evaluation only: choose lambda on the tuning images first (grid search with GRID_DC_LAMBDAS),
# then evaluate it here with --lambdas <chosen>,inf -- never pick lambda from this run.
#
# Submit:   sbatch run_soft_dc_test.sbatch --full-kspace                 (Run A, full k-space, job 98645)
#           sbatch run_soft_dc_test.sbatch                               (Run A, current/cropped pipeline)
#           sbatch run_soft_dc_test.sbatch --k 4 --init guided           (K-ensemble)
# Log:      slurm_logs/softdc-<jobid>.out (summary at the end)
# Results:  ~/fastmri_results/knee/diagnostics/soft_dc/<results csv name>/results.csv
# Preemption-safe: each finished image is appended to results.csv; a requeued job skips it.

#SBATCH --job-name=convdecoder_softdc
#SBATCH --output=slurm_logs/softdc-%j.out
#SBATCH --error=slurm_logs/softdc-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --account=acct-preempt
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00

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

echo "Job started on $(hostname) at $(date)"
python3 soft_dc_test.py "$@"
echo "Finished at $(date)."
