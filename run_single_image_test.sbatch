#!/bin/bash

# Fits a SINGLE image (Sections 7-11's demo: one full random-init reference fit, a second-image
# Run A/B/C comparison, and a results table) instead of the full 20/100-image batch pipeline
# (Section 12.5+) that run_knee_batch.sbatch/run_brain_batch.sbatch/run_notebook.sbatch run.
# Use this for a quick sanity-check run on an actual GPU compute node -- e.g. after the login
# node turned out to have a GPU (a GTX 1080 Ti, compute capability 6.1) too old for the installed
# PyTorch build to run on at all ("no kernel image is available for execution on the device").
#
# Review the result afterward by opening $OUT_NOTEBOOK (see below) in VS Code -- every plot
# (show_difference comparisons, the results table) is embedded in its saved cell outputs exactly
# as if you'd run the notebook interactively, plus tail slurm_logs/slurm-<jobid>.out for the
# live-streamed print()/tqdm output while it's still running.

# --- SLURM Settings ---
#SBATCH --job-name=convdecoder_single_image_test
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.out
#SBATCH --account=acct-preempt
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00               # <-- a single-image test is much shorter than the 100-
                                       #     image batch jobs' 24:00:00, but K_AVERAGING defaults
                                       #     to True (Section 2), so Section 8/12.1-12.4 still
                                       #     fit K_VALUE reference + accelerated checkpoints on
                                       #     top of Section 7's one full fit -- time a run once
                                       #     and adjust this down if it's overkill for your config.

# --- Setup Environment ---
mkdir -p slurm_logs outputs
source ~/.bashrc
source /home/itai.p/miniforge3/etc/profile.d/conda.sh
conda activate base

# --- Config overrides for this run ---
# CD_ANATOMY defaults to whatever Section 2 has hardcoded (currently "knee") if left unset here --
# uncomment to target brain instead:
#   export CD_ANATOMY=brain

# Headless run -- see Section 2's note on this toggle. Same as every other sbatch script.
export CD_EARLY_STOP_PLOT_LIVE=False

# The whole point of this script: skip Section 12.5+'s N-image batch loop even though
# K_AVERAGING itself stays on (so Section 8's Run A comparison and Section 12.1-12.4's
# single-image ConvDecoder-A ensemble result still run against the ONE image this job fits).
export CD_RUN_BATCH_EVAL=False

# Uncomment any of these to override Section 2's other defaults for this run without editing the
# notebook itself:
#   export CD_ARCHITECTURE=convdecoder_dcse
#   export CD_K_VALUE=4
#   export CD_RUN_MODE=new_fit
#   export CD_SEED=0
#   export CD_ACQUISITION_TYPE=CORPD_FBK
#   export CD_NUM_COILS=15

# --- Run Code ---
echo "Job started on $(hostname) at $(date)"

NOTEBOOK="MRI_ConvDeconv_variance_earlystop.ipynb"
OUT_DIR="outputs"
STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
FILTERED_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_${STAMP}_filtered.ipynb"
OUT_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_single_image_${STAMP}.ipynb"

# Strips ONLY "no-unattended" (Section 15's git-push cell) -- deliberately does NOT strip
# "skip-slurm", since that tag is exactly Sections 7-11's single-image demo, which is the whole
# point of this script. This is the mirror image of run_knee_batch.sbatch/run_brain_batch.sbatch/
# run_notebook.sbatch, which strip both tags to skip straight to the batch pipeline.
jupyter nbconvert \
    --to notebook \
    --TagRemovePreprocessor.remove_cell_tags='["no-unattended"]' \
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
echo "Output notebook (open in VS Code to see the reconstructions/plots/results table): $OUT_NOTEBOOK"
echo "Results CSV under: \${CONVDECODER_CKPT_ROOT:-<default CKPT_ROOT>}/results/"
