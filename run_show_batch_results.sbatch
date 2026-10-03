#!/bin/bash
# Figures of a few reconstructions from a finished batch run -- see show_batch_results.py. Reloads
# saved fits (no fitting): a few minutes on one GPU.
#
# Submit:   sbatch run_show_batch_results.sbatch                       (Run A)
#           sbatch run_show_batch_results.sbatch --k 4 --init guided   (K-ensemble)
#           sbatch --export=ALL,CD_READOUT_CROP=False run_show_batch_results.sbatch   (a full-k-space run,
#                                                                       once the notebook defaults to cropping)
# Log:      slurm_logs/show-<jobid>.out
# Figures:  ~/fastmri_results/knee/diagnostics/batch_examples/<results csv name>/*.png

#SBATCH --job-name=convdecoder_show
#SBATCH --output=slurm_logs/show-%j.out
#SBATCH --error=slurm_logs/show-%j.out
#SBATCH --account=acct-preempt
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

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
python3 show_batch_results.py "$@"
echo "Finished at $(date)."
