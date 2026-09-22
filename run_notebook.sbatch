#!/bin/bash

# --- SLURM Settings ---
#SBATCH --job-name=convdecoder_mri
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.out
#SBATCH --account=acct-preempt        # <-- fill in your actual account/partition/qos
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00               # <-- adjust to how long this run actually needs

# --- Setup Environment ---
mkdir -p slurm_logs outputs

# Load user settings -- match this to your actual conda install path (same idea as your
# train.sbatch example).
source ~/.bashrc
source /path/to/your/miniforge3/etc/profile.d/conda.sh
conda activate your-env-name

# --- Config overrides for this run (optional) ---
# Anything exported here overrides the matching CD_* toggle read by Section 2's "Environment-
# Variable Overrides" cell in the notebook, without editing the notebook itself. Leave a line
# commented out to just use whatever's hardcoded in Section 2.
#   export CD_ARCHITECTURE=convdecoder_dcse
#   export CD_K_VALUE=5
#   export CD_RUN_MODE=new_fit
#   export CD_SEED=0

# EARLY_STOP_PLOT_LIVE=False is important for a headless run: the live ipywidgets redraw mode
# (default, meant for an attached interactive kernel) produces nothing useful without one; the
# plain plt.show() fallback does, and lands in the executed notebook's saved cell outputs below.
export CD_EARLY_STOP_PLOT_LIVE=False

# --- Run Code ---
echo "Job started on $(hostname) at $(date)"

NOTEBOOK="MRI_ConvDeconv_variance_earlystop.ipynb"
OUT_DIR="outputs"
STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
FILTERED_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_${STAMP}_filtered.ipynb"
OUT_NOTEBOOK="${OUT_DIR}/${NOTEBOOK%.ipynb}_${STAMP}.ipynb"

# Step 1: strip cells tagged "skip-slurm" (Sections 7-11's single-image demo fits -- redundant
# with Section 12's own batch pipeline, which is what this generic full-run script is for) and
# "no-unattended" (Section 15's git push, which shouldn't run unattended inside a submitted job
# regardless of whether GitHub is even reachable from this node -- see Section 1's note on
# login-node-vs-compute-node internet access). This does NOT execute anything yet, just writes a
# filtered COPY with those cells removed; $NOTEBOOK itself is untouched. Want to run JUST the
# single-image demo instead of the full batch pipeline? Use run_single_image_test.sbatch, which
# keeps "skip-slurm" cells and strips only "no-unattended". To exclude more cells from SLURM runs
# later, tag them the same way in VS Code: select the cell -> "..." menu (or Cell actions) ->
# Add Cell Tag -> type the tag name.
jupyter nbconvert \
    --to notebook \
    --TagRemovePreprocessor.remove_cell_tags='["skip-slurm", "no-unattended"]' \
    --output "$(basename "$FILTERED_NOTEBOOK")" \
    --output-dir "$OUT_DIR" \
    "$NOTEBOOK"

# Step 2: execute the filtered notebook. papermill runs it top to bottom like `nbconvert
# --execute` would, but ALSO streams each cell's stdout (every print(), including tqdm progress
# and the [env override]/[GUI] confirmation lines) live into THIS script's own stdout via
# --log-output -- which is what lands in slurm-%j.out, so you can `tail -f
# slurm_logs/slurm-<jobid>.out` while it's still running. Every matplotlib figure (plt.show()
# calls -- see EARLY_STOP_PLOT_LIVE above) still gets captured as an embedded image in the SAVED
# output notebook, not the plain-text SLURM log (images can't go in a text log) -- open
# $OUT_NOTEBOOK afterward in VS Code to see them exactly as if you'd run the notebook
# interactively.
#
# Needs `pip install papermill` once in your conda env. If you'd rather not add that dependency,
# swap this step for:
#   jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
#       --output "$(basename "$OUT_NOTEBOOK")" --output-dir "$OUT_DIR" "$FILTERED_NOTEBOOK"
# -- functionally equivalent (same saved-output-notebook result with all plots/prints embedded),
# just without the live-streamed-to-the-SLURM-log part.
papermill \
    "$FILTERED_NOTEBOOK" \
    "$OUT_NOTEBOOK" \
    --log-output \
    --execution-timeout -1

rm -f "$FILTERED_NOTEBOOK"  # intermediate only -- $OUT_NOTEBOOK is the real saved result

echo "Finished at $(date)."
echo "Output notebook (all plots + printed metrics embedded, open in VS Code): $OUT_NOTEBOOK"
