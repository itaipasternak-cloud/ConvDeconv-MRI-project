#!/bin/bash
# Refit a few grid-search tuning images under several (huber_delta, tv_weight) settings and save
# comparison figures -- see view_grid_settings.py's docstring. Default: 3 images x 4 settings =
# 12 fits of 6000 iterations, roughly 3h (grid fits averaged ~14 min each).
#
# Submit (extra arguments are passed straight to view_grid_settings.py):
#   sbatch run_view_grid_settings.sbatch
#   sbatch run_view_grid_settings.sbatch --settings 0.3:1.2e-3,0.3:4.75e-4 --n-images 5
# Progress: tail -f slurm_logs/view-<jobid>.out
# Figures:  ~/fastmri_results/knee/grid_search/viewer/*.png (open them in VS Code)
#
# Preemption-safe: each finished fit is cached, and masks are seeded from the filename, so a
# requeued job redoes only the fit in progress.

#SBATCH --job-name=convdecoder_view
#SBATCH --output=slurm_logs/view-%j.out
#SBATCH --error=slurm_logs/view-%j.out
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
export CD_EARLY_STOP_PLOT=False
export CD_EARLY_STOP_PLOT_LIVE=False
export CD_K_AVERAGING=False
export CD_RUN_BATCH_EVAL=False
export CD_RUN_GRID_SEARCH=False
export TQDM_MININTERVAL=60     # one progress line a minute in the log, not thousands

echo "Job started on $(hostname) at $(date)"
python3 view_grid_settings.py "$@"
echo "Finished at $(date)."
