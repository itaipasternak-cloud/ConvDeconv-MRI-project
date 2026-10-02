#!/bin/bash
# Is the MRI-VAE latent normally distributed? Encodes the 30 grid-search tuning images with the
# frozen microsoft/mri-autoencoder-v0.1 (Z_SOURCE="mri_vae"), no fitting -- see
# vae_latent_distribution.py. About 10 minutes on one GPU.
#
# Submit:   sbatch run_vae_latent_check.sbatch          (extra args go to the script, e.g. --n-images 10)
# Log:      slurm_logs/vae-<jobid>.out (statistics table at the end)
# Figures:  ~/fastmri_results/knee/diagnostics/vae_latent/*.png
#
# The first run installs diffusers and downloads the VAE from Hugging Face (Section 4 of the
# notebook). If the compute node has no internet, run these once on the login node first:
#   pip install "diffusers>=0.24.0" "huggingface_hub>=0.20.0" safetensors
#   python3 -c "from diffusers.models import AutoencoderKL; AutoencoderKL.from_pretrained('microsoft/mri-autoencoder-v0.1')"

#SBATCH --job-name=convdecoder_vae_check
#SBATCH --output=slurm_logs/vae-%j.out
#SBATCH --error=slurm_logs/vae-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --account=acct-preempt
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

mkdir -p slurm_logs
source ~/.bashrc
source /home/itai.p/miniforge3/etc/profile.d/conda.sh
conda activate base

export CD_ANATOMY=knee
export CONVDECODER_DATA_DIR=~/fastmri_data/knee_multicoil_combined
export CONVDECODER_CKPT_ROOT=~/fastmri_results/knee
export CD_Z_SOURCE=mri_vae
export CD_K_AVERAGING=False
export CD_RUN_BATCH_EVAL=False
export CD_RUN_GRID_SEARCH=False
export CD_EARLY_STOP_PLOT=False
export CD_EARLY_STOP_PLOT_LIVE=False

echo "Job started on $(hostname) at $(date)"
python3 vae_latent_distribution.py "$@"
echo "Finished at $(date)."
