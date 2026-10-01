# CLAUDE.md

Research project by Itai Pasternak, built on top of a fork of the ConvDecoder paper repo
("Accelerated MRI with Un-trained Neural Networks", Darestani & Heckel). `README.md` is the
**original paper's** README and does not describe this project's own work -- this file does.

## What the project is

Un-trained (DIP-style) reconstruction of undersampled multicoil fastMRI data (knee, and brain
set up but not yet run at scale). On top of the original ConvDecoder it adds:

- **Architecture**: `convdecoder_dcse` (current default) -- squeeze-excitation blocks whose gate
  is conditioned on the data-consistency residual (`DCSEBlock`). Also `convdecoder`, `convdecoder_se`.
  Optional `FreqRegBlock` (frequency regularization ramp) and `KaiserUpsample`, both off.
- **Loss**: Huber data-fidelity + Total Variation (`LOSS_TYPE="huber"`, `HUBER_DELTA`, `TV_WEIGHT`).
- **Ground-truth-free early stopping**: `EARLY_STOP_METHOD="variance"` (running variance of the
  network output) or `"holdout"` (held-out k-space slice).
- **ConvDecoder-A ensemble**: `K_VALUE` accelerated fits (`ACCEL_NUM_ITERS`), each guided-init
  from a *different* reference image's converged checkpoint, then (weighted, complex) averaged.
  Evaluated in batch over up to 100 images for paper-style mean ± std.
- Input-code options (`Z_SOURCE`: uniform / uniform_blur / gaussian / mri_vae), uncertainty maps,
  ESPIRiT sensitivity maps with fixed ACS width, Optuna + exhaustive grid hyperparameter search.

Metrics everywhere: PSNR, SSIM, MS-SSIM, VIF (higher better), NMSE, HFEN (lower better).
The standard columns compare normalize(gt) with normalize(rec) (each divided by its OWN max), which
penalizes a bright artifact pixel or extra noise everywhere. Since 2026-10-01 the grid search and
batch also record `*_raw` (no rescaling -- rec is already on gt's scale), `*_ls` (least-squares
brightness match) and `PSNR_fg`/`SSIM_fg` (raw, anatomy only); see `compute_metric_variants()`.
Which convention the paper reports is still to be decided (being compared with view_grid_settings.py).

## Where things live

- **`MRI_ConvDeconv_variance_earlystop.ipynb` -- the main and only active notebook.** All work
  happens here. Section map is in its first markdown cell. Key cells:
  - Section 1: setup. Off-Colab it runs `git pull` in the repo dir and sets git identity.
  - **Section 2: all config toggles** (one huge cell), `RUN_TAG` derivation, paths,
    checkpoint guards, and the `CD_*` environment-variable override loop.
  - Section 3: `build_network()` + architecture classes. Section 5.1: the notebook's own `fit()`
    (overrides `demo_helper/fit_multicoil.fit`) with Huber/TV, early stopping, LR schedule.
  - Section 6.5: hyperparameter grid search (5x5 HUBER_DELTA x TV_WEIGHT, 30 images, Run A regime),
    sharded across a SLURM array; results in `CKPT_ROOT/grid_search/<GRID_SEARCH_NAME>/`.
  - Section 12 / 12.5: K-ensemble single image, then the real batch pipeline.
- `MRI_ConvDeconv_espirit.ipynb`, `*.ipynb.bak`, `ConvDecoder_for_MRI.ipynb`: older versions, not
  maintained. The other notebooks (`ConvDecoder_vs_*`, `robustness_*`, `visualize_*`) are the
  original paper's.
- `demo_helper/` (helpers, `fit_multicoil.py`), `include/`, `common/`, `DIP_UNET_models/`: library
  code, mostly from the original repo (lightly modernized June 2026).
- `run_*.sbatch`: SLURM jobs (see below). `analyze_grid_search.py`: ranks a finished grid search
  and picks the winner (scoring weights live here). `view_grid_settings.py` (+
  `run_view_grid_settings.sbatch`): refits a few tuning images under chosen settings and saves
  comparison figures + full-frame/ROI metrics. `check_image_budget.py`: free images per
  acquisition/coil group after existing selections. `check_*_pool.py`,
  `setup_knee_combined_pool.sh`: dataset pool checks/setup.
- `outputs/`, `slurm_logs/`, `9029*.out`: a few committed job outputs/logs (early runs).

## How it runs

**Compute is a SLURM cluster, not this laptop.** Host `ece-silbmark2`, user `itai.p`,
`--account=acct-preempt --partition=part-preempt --qos=qos-preempt`, 1 GPU, conda at
`/home/itai.p/miniforge3` (env `base`). Jobs are preemptible: every script uses `--requeue` +
`--open-mode=append`, and the pipeline resumes from cached checkpoints / results CSVs.
Colab (Google Drive paths) is still supported by the notebook but is no longer the main path.

Every sbatch script does the same thing: `jupyter nbconvert` strips cells tagged `skip-slurm`
(Sections 7-11 single-image demo) and `no-unattended` (Section 2.1 config GUI, Section 15 git push),
then `papermill`
executes the notebook headless into `outputs/`. Config is changed per job via `CD_<NAME>`
environment variables (e.g. `CD_K_VALUE=1`), registered in Section 2's override loop -- add a
new toggle to that list if a job needs to override it.

Cluster paths: data `~/fastmri_data/knee_multicoil_combined` (val + train_batch_0 symlinked,
needed for 100 eligible images per acquisition group); results/checkpoints
`~/fastmri_results/knee` (`CONVDECODER_CKPT_ROOT`) with `results/`, `checkpoints/`,
`selections/`, `tuning/` under it.

Main scripts:
- `run_knee_batch.sbatch [N]` -- the real K-ensemble batch run (default 100 images; pass `1` for a
  quick test). ~12.5 min/image with cached references.
- `run_knee_batch_runA.sbatch [N]` -- paired baseline: same 100 images, K=1, random init,
  6000 iterations.
- `run_single_image_test.sbatch` -- single-image sanity check (keeps Sections 7-11).
- `run_brain_batch.sbatch` -- brain version (time limit not yet calibrated).
- `run_knee_grid_search.sbatch` -- the grid search as one SLURM array (12 tasks, max 12 at once --
  the cluster is shared, keep it under ~10-15 simultaneous jobs). ~175 GPU-hours, ~15h per task.
  Then `python3 analyze_grid_search.py --grid-dir ~/fastmri_results/knee/grid_search/grid_v2 --plot`.
  A failed shard can be rerun split over several GPUs, skipping fits already recorded:
  `sbatch --array=0-3 --export=ALL,RERUN_SHARD=5,RERUN_PARTS=4 run_knee_grid_search.sbatch`.
  Each search's name/grid/image reuse is set in Section 2 (`GRID_SEARCH_NAME`, `GRID_REUSE_FROM`, ...).

## Current state (as of 2026-10-01)

- Default config: `convdecoder_dcse`, uniform z, Huber+TV, variance early stopping, LR schedule on,
  `K_VALUE=4`, guided init from different images, `ACCEL_NUM_ITERS=1350`, `SEED=0`.
- **Previous grid search** (g1-g5 + u1-u14, 81 combos, 13 images, composite score, scored
  WITHOUT data consistency and with a different seed than the batch) picked `HUBER_DELTA=0.36`,
  `TV_WEIGHT=3.5e-05`, still the Section 2 defaults (one set for all regimes;
  `USE_SEPARATE_ACCEL_PARAMS=False` so `ACCEL_*` mirror them).
- `KAVG_BATCH_MANUAL_EXCLUDE` holds 6 knee files dropped from the batch on purpose.
- **The grid winner made the 100-image batch results worse**: Run A 32.3 dB, K-ensemble 33.3 dB
  PSNR. Replaced by the rebuilt grid search (Section 6.5, `grid_v2`): same fit/scoring path as the
  batch, per-image metrics saved, winner chosen by `analyze_grid_search.py`. Submitted
  2026-09-28 as array job 96483 (logs `slurm_logs/grid-96483_<0-11>.out`); shards 0-4 and 6-11
  finished, shard 5 died before its first fit (papermill IOPub timeout in the GUI cell) and is
  rerun in 4 parts (job 96813). grid_v2 result: top ~12 of 25 statistically tied within ~0.4 dB
  PSNR (~32.6-33.0 dB); best at the upper edges (delta 0.65, tv 7.5e-5). Old default ~(0.14,
  1.2e-5) = 32.70 dB, old winner ~(0.3, 3e-5) = 32.47 dB -- confirms the old winner was worse.
  Follow-up grid_v3: delta [0.65, 1.3, 2.6, 100 (=MSE)] x tv [3e-5, 7.5e-5, 1.9e-4], same 30
  images, 2 shared pairs copied from grid_v2 (GRID_REUSE_FROM). grid_v3 result (job 97018):
  PSNR falls steadily with delta (100 = plain MSE is ~2 dB worse, HFEN much worse) -> delta ~0.65
  is the optimum and Huber clearly helps. Best of the 35 settings: (0.65, 1.9e-4), PSNR 32.92 dB
  (tied best), SSIM 0.864 (best), vs old default 32.70 / 0.847 -- but on the TV edge.
  grid_v4: delta [0.3, 0.65, 1.3] x tv [4.75e-4, 1.2e-3], same images. Ranked together (41
  settings, 2026-10-01): winner (0.3, 1.2e-3) PSNR 33.46 / SSIM 0.876, statistically tied with
  (0.3, 4.75e-4) 33.50 / 0.871 and (0.65, 1.2e-3) 33.20 / 0.874; on the grid edge for both delta
  (0.14 untested at high TV) and TV. At delta 0.3 PSNR has flattened while SSIM still rises; VIF
  falls as TV rises (possible over-smoothing), and full-frame metrics also reward flattened
  background. `view_grid_settings.py` (job 98026) refits 3 tuning images x 4 settings on shared
  masks to look at this. Then: pick the settings (maybe grid_v5 on seeded masks), set them in
  Section 2, run the 100-image Run A + K-ensemble batches.
- **Undersampling masks are seeded per image since 2026-10-01** (`mask_seed_for()`, Section 5:
  SEED + checksum of the image's k-space). Before that, MaskFunc reseeded from OS entropy on
  every call, so grid_v2-v4 compared settings on different masks (unbiased, but noisier -- their
  ties/SEs already include that noise), Run A and the K-ensemble used different masks, and
  cached fits reloaded after a requeue got data consistency from the wrong mask. Cached batch
  members are now refit if their saved mask differs (`cached_fit_mask_matches()`), and the grid
  config records `mask_seeding`, so new searches don't import/rank with grid_v2-v4.
- The two old 100-image lists (keyed by 0.1192 and 0.36 RUN_TAGs) contain DIFFERENT images, so
  the earlier before/after batch comparison was on different image sets. The next batch run will
  stop until one is copied to `selections/kavg_batch_selection_n100.json` (plan: the 0.1192 one,
  if the pre-grid results were run on it). A separate set of
  values for the K-ensemble regime is planned later.
- Image budget: ~200 knee CORPD_FBK/15-coil files (val + train_batch_0). One experiment per
  group uses ~141 (7 demo/default + 4 refs + 100 batch + 30 tuning). Brain AXT2 4-coil (165) and
  16-coil (161) groups are enough; smaller groups are not.
- **Compute budget rule (r ≈ 1.3)**: one DCSE iteration costs ~1.3x a vanilla ConvDecoder
  iteration (measured by Section 12.7). Total compute must stay at or below the original
  method's, which is why Run A uses 6000 iterations (roughly 9000 vanilla-equivalent, per Itai).
  Don't raise iteration counts without checking this.
- Open question: whether the brain run is still planned.

## Rules and conventions

- **Pushing to `master` changes what cluster jobs run.** Section 1 runs `git pull` when a job
  starts, so a queued job picks up whatever is on `master` at that moment. Don't push half-done
  notebook changes while jobs are queued.
- **Never re-enable the notebook's auto git push** (Section 15). It once pushed to `master`
  unasked and collided with other work. Commit and push manually.
- **Do not commit** checkpoints, large results, or `download_*.sbatch` (contain presigned AWS URLs;
  the repo is public). See `.gitignore`.
- Keep the notebook committed **with outputs stripped** and valid JSON. Prefer editing it with a
  notebook-aware tool, not raw text edits.
- **`RUN_TAG` naming is load-bearing**: checkpoint/results paths are derived from config values,
  and `load_checkpoint_guarded()` / `assert_run_tag_current()` refuse mismatches. Any new setting
  that changes the fit result must be added to `checkpoint_metadata()` (and to `RUN_TAG` if it
  changes the network structure), keeping old checkpoints loadable.
- Anything computed from config values must be computed *after* the `CD_*` override loop, or
  overrides are silently ignored (this bug has happened more than once).
- Parallel jobs must write to distinct files (e.g. `results_shard<i>.csv`), then merge
  afterwards -- never share one output file between concurrent jobs.
- **Image selections are fixed and must not depend on hyperparameters**: the 100-image batch
  (`selections/kavg_batch_selection_n100.json`) and references (`kavg_ref_selection_kavg<K>.json`)
  are keyed only by size/K. Tuning images are never drawn from them. Change them only on purpose.
- Style: long explanatory comments that give the *why* next to each toggle, and detailed commit
  messages that say what changed, why, and which job ID exposed it. Match this.
- Markdown cells in the notebook are sometimes stale (e.g. "k=10", "N=20"); the code in Section 2
  is the source of truth.
