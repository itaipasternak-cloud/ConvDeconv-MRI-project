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
**All metrics are on the raw intensity scale** (`to_metric_scale()`, decided 2026-10-02): the
reconstruction is NOT rescaled; gt and rec are both divided by gt's max (rec is already on gt's
scale -- reconstruct() undoes scaling_factor). The old convention, normalize() = each image divided
by its OWN max, penalized one bright artifact pixel or extra noise everywhere (a fit scored 22.3 dB
that way vs 31.2 dB raw). Anatomy-only `PSNR_fg`/`SSIM_fg`
(`anatomy_metrics()`) are still recorded in the CSVs but are **not reported in the paper** (Itai,
2026-10-03): the paper reports the whole-image metrics on the central 320 x 320. grid_v2-v4 results
are max-normalized only; grid_v5's standard columns are max-normalized with `*_raw`/`*_ls`
alongside; searches after it have `metric_scale: raw` in their manifest.

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
- `MRI_ConvDeconv_espirit.ipynb`, `ConvDecoder_for_MRI.ipynb`: older versions, not
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
- `outputs/`, `slurm_logs/`: a few committed job outputs/logs (early runs).
- `tests/e2e_cpu/`: CPU end-to-end test of the notebook on synthetic data (see its README).
- `soft_dc_test.py` (+ `run_soft_dc_test.sbatch`): soft vs hard data consistency, re-scoring a
  finished batch run's saved fits for lambda in {0 (no DC), 0.03 ... 10, inf (hard)} -- no fitting;
  paired differences to hard DC with Wilcoxon p-values. `--full-kspace` for runs made before
  readout cropping (e.g. Run A job 98645), `--k 4 --init guided` for the K-ensemble.
- `show_batch_results.py` (+ `run_show_batch_results.sbatch`): figures of a finished batch run's
  reconstructions (best / median / worst by PSNR, or chosen images), reloaded from the saved
  member fits -- no fitting; also re-checks the metrics against the run's CSV.
- `readout_crop_test.py` (+ `run_readout_crop_test.sbatch`): fit full (640 x 368) vs
  readout-cropped (320 x 368, the readout oversampling removed exactly) k-space on 8 tuning images
  x 3 TV weights, same masks/scoring; quality, iterations and time per iteration.
- `vae_latent_distribution.py` (+ `run_vae_latent_check.sbatch`): is the MRI-VAE latent (Z_SOURCE
  "mri_vae") normally distributed? Encodes the 30 tuning images, per-channel + pooled stats,
  histograms/Q-Q plots in `CKPT_ROOT/diagnostics/vae_latent/`.

## How it runs

**Compute is a SLURM cluster, not this laptop.** Host `ece-silbmark2`, user `itai.p`,
`--account=acct-preempt --partition=part-preempt --qos=qos-preempt`, 1 GPU, conda at
`/home/itai.p/miniforge3` (env `base`). Jobs are preemptible: every script uses `--requeue` +
`--open-mode=append`, and the pipeline resumes from cached checkpoints / results CSVs.
Colab (Google Drive paths) is still supported by the notebook but is no longer the main path.

Every sbatch script does the same thing: `jupyter nbconvert` strips cells tagged `skip-slurm`
(Sections 7-11 and 12.2-12.4, the single-image demos) and `no-unattended` (Section 2.1 config GUI,
Section 15 git push) -- this needs `--TagRemovePreprocessor.enabled=True`, which was missing until
2026-10-02, so every earlier job (all batches, grid_v2-v5) ran those cells too, including Section
7's 10,000-iteration demo fit. Then `papermill`
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

## Current state (as of 2026-10-02)

- Default config: `convdecoder_dcse`, uniform z, Huber+TV, variance early stopping, LR schedule on,
  `K_VALUE=4`, guided init from different images, `ACCEL_NUM_ITERS=1350`, `SEED=0`.
- **Section 2 defaults since 2026-10-02: `HUBER_DELTA=0.14`, `TV_WEIGHT=1.2e-3` (grid_v5's
  winner)**, one set for all regimes (`USE_SEPARATE_ACCEL_PARAMS=False`, so `ACCEL_*` mirror them)
  until the accelerated regime gets its own grid search.
- **Previous grid search** (g1-g5 + u1-u14, 81 combos, 13 images, composite score, scored
  WITHOUT data consistency and with a different seed than the batch) picked `HUBER_DELTA=0.36`,
  `TV_WEIGHT=3.5e-05`.
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
  masks: under raw-scale metrics (0.3, 1.2e-3) was best or tied on all 3 images, and the
  max-normalized metric turned out very noisy (one fit: 22.3 dB max-normalized vs 31.2 dB raw),
  so grid_v2-v4's fine ranking is unreliable.
  **grid_v5** = the full grid, delta [0.03 ... 2.6, 100] x TV [2e-6 ... 7.5e-3] (80 settings x
  30 images = 2400 fits, same tuning images as grid_v2), seeded masks, raw metrics recorded. Rank
  with `analyze_grid_search.py --grid-dir .../grid_v5 --scale raw --plot`. **Result (job 98181):
  winner (0.14, 1.2e-3), PSNR 34.19 dB / SSIM 0.876** (raw), tied with (0.065, 1.2e-3), (0.14,
  4.75e-4), (0.3, 1.2e-3), (0.65, 1.2e-3); (0.3, 1.2e-3) had the top score/PSNR (34.26) but failed
  the SSIM guard. TV matters, delta doesn't: TV 1.2e-3 is best for every delta 0.065-0.65 and is
  inside the grid; delta 0.065-0.65 changes PSNR < 0.15 dB; MSE (delta 100) is ~2.3 dB worse. VIF
  falls as TV rises (0.80 at the winner vs ~0.86 at low TV) while HFEN improves. +1.17 dB / +0.034
  SSIM over the original default (0.14, 1.2e-5).
- Fixed evaluation set (2026-10-01): the delta-0.36 run's lists were copied to
  `selections/kavg_batch_selection_n100.json`, `kavg_ref_selection_kavg4.json`,
  `kavg_ref_selection_kavg1.json`, and `kavg_batch_selection_n1.json` = its first image (so a
  1-image test run accumulates into the 100-image results). The delta-0.1192 list was not usable:
  it contains 6 tuning images.
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
- **Every slow, full fit is at most 6000 iterations** (2026-10-03): Run A, the grid searches, and
  the K-ensemble's reference fits (`num_iters_slow = 6000`, was 10000; `num_iters_A = 6000`). A saved
  reference fit with another count is refit, not reused (`reference_matches_settings()`); the
  full-k-space references in `..._refs` (without `_rocrop`) are 10000-iteration ones. Known gap:
  ensemble MEMBER checkpoints don't record which reference version they started from, so members
  saved before their references were refit would be reused -- delete them if that ever happens.
- **Compute budget rule (r ≈ 1.3)**: one DCSE iteration costs ~1.3x a vanilla ConvDecoder
  iteration (measured by Section 12.7). Total compute must stay at or below the original
  method's, which is why Run A uses 6000 iterations (roughly 9000 vanilla-equivalent, per Itai).
  Don't raise iteration counts without checking this.

## Goals and roadmap (as of 2026-10-02)

Goal: ONE paper accepted in a Q1/Q2 journal, then the thesis built on it. Choose the
methodologically right option over matching how the original ConvDecoder paper did things. Don't
draft paper text until Itai asks.

**Paper scope:** training-free + ground-truth-free (settings from the data -- delta from the noise
level, TV by the discrepancy principle -- validated against grid_v5 as the oracle), compute-matched,
calibrated uncertainty (conformal, AUSE, fastMRI+ pathology), high-field -> low-field (incl.
high-field references guiding low-field fits); with baselines (CS L1-wavelet, original
ConvDecoder, DIP/Deep Decoder, pretrained fastMRI U-Net/VarNet), ablations, knee + brain, 4x and
8x, low-field data.

Done:
- [x] Mask seeding per image; raw-scale metrics as the standard; anatomy-only metrics.
- [x] grid_v5 (Run A regime, full 8x10 grid): winner HUBER_DELTA=0.14, TV_WEIGHT=1.2e-3, set in
      Section 2.
- [x] Batch rows record iterations and fit time actually spent (equal-compute claim).
- [x] SLURM cell stripping fixed (it had never worked); `import pandas` moved to Section 1 (the batch
      would otherwise have crashed); derived config recomputed after the `CD_*` overrides
      (`_derive_config()`); Run A pinned to the regular settings (`CD_USE_SEPARATE_ACCEL_PARAMS=False`).
- [x] Accelerated grid search (`GRID_REGIME="accel"`): same 80 settings x 30 images as grid_v5,
      each scored on the 4-member ensemble average; references at the regular settings, fit once
      in parallel by the array tasks via `fit_reference()` (shared with Section 12.1).
- [x] CPU end-to-end test (`tests/e2e_cpu/`, 640x368 synthetic data).

- [x] **Readout cropping adopted (2026-10-02):** `READOUT_CROP=True` (Section 2) removes the 2x
      readout oversampling at every k-space load (`maybe_readout_crop()`, Section 5): 640x368 ->
      320x368, 1.64x faster per iteration, ~32% less time per fit -- at a small, consistent quality
      cost on the crop test (job 98647, 8 tuning images, paired): at TV 1.2e-3 PSNR -0.15 +/- 0.08
      dB, SSIM +0.0015 +/- 0.0012, VIF 0.780 vs 0.804, HFEN 0.341 vs 0.329; PSNR also lower at TV
      4.75e-4 (-0.33) and 3e-3 (-0.24); best TV unchanged. Adopted for the time saving; the
      100-image cropped vs full Run A comparison is the definitive check (switch back before the
      K-ensemble batch if the cost is clearly larger). Tag `_rocrop` keeps it apart from all
      earlier full-k-space results (grid_v2-v5, Run A job 98645, grid_accel_v1).
- [x] **grid_accel_v1, full k-space (jobs 98567 + 98646, 2400 ensembles):** accelerated winner
      **delta 0.065, TV 4.75e-4** -- ensemble PSNR 34.61 dB, SSIM 0.879 on the 30 tuning images; tied
      only with (0.03, 1.9e-4); interior on both axes. The short fits want gentler settings than Run
      A (0.14, 1.2e-3): averaging already denoises. The ensemble at Run A's settings scores 34.43 /
      0.876, so separate accelerated settings are worth ~0.18 dB. MSE (delta 100) ~1.3 dB worse.
      Ensemble vs Run A at each one's best settings, same 30 images/masks/metrics: +0.42 dB, +0.003
      SSIM (grid_v5 Run A: 34.19 / 0.876) -- real but well below the old ~1 dB; compare at actual
      compute spent.
- [x] **Run A, full k-space, 100 images (job 98645):** PSNR 34.00 +/- 2.36 dB, SSIM 0.871 +/- 0.039,
      MS-SSIM 0.968, VIF 0.820, NMSE 0.0051, HFEN 0.325, knee-only 32.95 dB / 0.849; 3350 +/- 1935
      of 6000 iterations actually run, 122 s per image. (grid_v5 predicted 34.19 / 0.876 for these
      settings on the 30 tuning images -- the tuning carried over.)

Next, in order (all on the cropped pipeline):
1. **grid_crop_v1** (job 98976; Run A regime, delta [0.065, 0.14, 0.3] x TV [4.75e-4, 1.2e-3, 3e-3],
   same 30 images) confirms or moves the regular winner (0.14, 1.2e-3); set it in Section 2 if it moves.
2. Then submit together (Section 2 switched to the accelerated confirmation grid first):
   a. **Run A cropped, 100 images** (`run_knee_batch_runA.sbatch`; hard DC) -- final-pipeline Run A,
      and the 100-image cropped-vs-full comparison (keep cropping only if VIF/HFEN don't drop
      clearly, e.g. VIF by >= 0.02).
   b. **Accelerated confirmation grid, cropped, with soft-DC scoring:** GRID_SEARCH_NAME
      "grid_accel_crop_v1", GRID_REGIME "accel", delta [0.03, 0.065, 0.14] x TV [1.9e-4, 4.75e-4,
      1.2e-3], GRID_DC_LAMBDAS [0, 0.3, 1, 3, 10, 30]; `sbatch --array=0-3 --export=ALL,GRID_NUM_SHARDS=4
      run_knee_grid_search.sbatch` (~15 GPU-hours). Fits the 4 cropped references (6000 iter) first.
   c. **Run A-regime lambda run:** the regular winner alone on the 30 tuning images with soft-DC
      scoring, e.g. `sbatch --array=0 --export=ALL,GRID_NUM_SHARDS=1,CD_GRID_SEARCH_NAME=grid_crop_dc_v1,
      CD_GRID_REGIME=run_a,CD_GRID_HUBER_DELTA_VALUES=0.14,CD_GRID_TV_WEIGHT_VALUES=0.0012,
      CD_GRID_DC_LAMBDAS='0;0.3;1;3;10;30' run_knee_grid_search.sbatch` (~1 GPU-hour).
3. Decisions from 2: keep cropping? accelerated delta/TV (`USE_SEPARATE_ACCEL_PARAMS=True` +
   `ACCEL_*`); lambda for Run A and for the ensemble -- all chosen on the tuning images
   (`analyze_grid_search.py` ranks delta, TV and lambda jointly).
4. K-ensemble 1-image test, then the **K-ensemble 100-image batch**. Its fits don't depend on lambda
   (DC is applied only when reconstructing); it is scored with hard DC and re-scored with the chosen
   lambda without refitting.
5. Paired analysis script: Run A vs ensemble per image (mean +/- SE, Wilcoxon), compute actually
   spent. Soft-vs-hard DC on the 100 images with `soft_dc_test.py --lambdas <chosen>,inf` --
   EVALUATION of the lambda chosen in 3, never used to choose it.
6. Reference-settings sensitivity check (`GRID_REF_HUBER_DELTA`/`GRID_REF_TV_WEIGHT`).
7. Baselines; ablation table (DCSE, Huber+TV, early stopping on/off, ensemble, guided init,
   automatic vs grid-searched settings, readout cropping, soft vs hard DC); 8x; brain; uncertainty
   calibration; ground-truth-free settings (incl. lambda from the noise level); low-field (M4Raw).

**Methodology rule:** every setting (delta, TV, lambda, accelerated settings, ...) is chosen on the 30
tuning images only; the 100-image evaluation set is used once, with everything fixed in advance.

Soft vs hard DC: `reconstruct(..., dc_lambda=...)` / `apply_data_consistency()` -- None = hard (the
default: measurement replaces the prediction at acquired locations, re-inserting its noise), a
number = soft, k = (lambda * measured + predicted) / (1 + lambda) (0 = no DC). Grid searches score
every fit under GRID_DC_LAMBDAS as extra `<metric>_dc<lambda>` columns (no extra fitting).

Decisions and open questions:
- Speed over bit-for-bit reproducibility: `cudnn.benchmark = True` stays, so refitting the same
  config can stop at a different iteration and differ slightly (seen in jobs 98026/98111).
- Early stopping (variance, patience 2, 12% margin) stops fits anywhere from ~1500 to 6000 of 6000
  iterations, and the stopping point moves a fit's PSNR by up to ~1 dB. Whether it helps vs. a
  fixed budget is untested -- in the ablation.
- `KAVG_WEIGHTED_ENSEMBLE=True` with `KAVG_WEIGHT_EXPONENT=0` is a plain (uniform) average.

## Notes for the paper (briefing for whoever writes it -- notes, not drafted text)

Don't draft until Itai asks. One article (Q1/Q2), knee first; brain, more accelerations, uncertainty
maps and low-field are part of the plan. All numbers below are provisional -- take final numbers
from the final runs' CSVs (`CKPT_ROOT/results/`, `CKPT_ROOT/grid_search/*/summary_*.csv`).

**Story / claims.** Un-trained (no training data), compute-matched accelerated MRI reconstruction
with a K-ensemble, calibrated uncertainty, carrying over from high- to low-field. Be precise about
what "training-free" means: no network is trained on a dataset, BUT delta/TV/lambda were tuned with
ground truth on 30 tuning images -- disclose this, unless the ground-truth-free settings (delta
from the noise level, TV by the discrepancy principle, lambda* = sigma_p^2/sigma_n^2) are done
and validated against the grid-search oracle, which would make the claim fully true.

**Method, exact settings (as of 2026-10-03; re-check Section 2 before writing).**
- Network: ConvDecoder, 7 stages, 256 channels; fixed input z ~ Uniform(0,1), shape 256 x 8 x 4;
  each stage nearest-neighbor upsample (geometric size schedule to the k-space grid) -> 3x3 conv ->
  ReLU -> BatchNorm -> DCSE; final 1x1 conv to 2 x coils (real/imag per coil).
- DCSE (the architectural contribution): squeeze-excitation (reduction 16, hidden = max(C/16, 4))
  whose gating MLP also receives a scalar data-consistency signal -- the previous iteration's
  data-fidelity loss (detached). Compare against plain SE and no SE in the ablation.
- Forward model: per-coil FFT of the output, multiplied by the undersampling mask, compared with
  the measured k-space scaled by an RMS-matching scaling factor (pred vs target RMS on acquired
  samples, computed once before fitting).
- Loss: Huber(delta) data fidelity + TV_WEIGHT * TV, TV = mean |vertical diff| + mean |horizontal
  diff| on the raw per-coil real/imag output. Adam lr 0.01, cosine decay to 0.005 over the budget.
- Early stopping (ground-truth-free): every 150 iterations, variance of the network output across a
  window of the last 4 checks; stop when it exceeds 1.12 x its minimum for 2 consecutive checks
  (not before 20% of the budget); keep the latest checkpoint within the margin. Reported compute =
  iterations actually run.
- Reconstruction: data consistency at reconstruction time only (hard, or soft with lambda --
  formula below), ESPIRiT coil combination (sigpy EspiritCalib, 23-column calibration window from
  the fully sampled center; maps estimated from the undersampled data), 320 x 320 center crop.
- Run A (single-fit baseline): random init (seed SEED+1000), <= 6000 iterations, regular settings.
- K-ensemble (ConvDecoder-A): K = 4 members; member i starts from the converged weights of a
  random-init fit (<= 6000 iterations, regular settings) of a DIFFERENT image (4 fixed reference
  images, excluded from tuning and evaluation), gets a NEW random z (seed SEED+2000+i), and is fit
  <= 1350 iterations on the target with the accelerated settings; members averaged in the complex
  domain with uniform weights (KAVG_WEIGHTED_ENSEMBLE=True with exponent 0 is uniform -- describe it
  as a plain average). Reference fits are made once and reused for all images: report their
  one-time cost separately (4 x <= 6000 iterations), amortized per image.
- Readout cropping: the 2x readout oversampling is removed before fitting (exact; inverse FFT along
  readout, keep the central half, FFT back): 640 x 368 -> 320 x 368.
- Compute rule: one DCSE iteration ~1.3x a vanilla ConvDecoder iteration (Section 12.7);
  budgets were set so total compute stays <= the original method's.
- Soft data consistency: for coil c at k-space location r, with network prediction k_hat_c(r),
  measurement y_c(r) (times the scaling factor) and acquired set Omega:
  k_out_c(r) = (lambda * y_c(r) + k_hat_c(r)) / (1 + lambda) for r in Omega, k_hat_c(r) otherwise;
  lambda = 0 is no DC, lambda -> infinity hard DC. At each acquired location this minimizes
  |k - k_hat_c(r)|^2 + lambda * |k - y_c(r)|^2; with measurement-noise variance sigma_n^2 and
  prediction-error variance sigma_p^2 the optimal weight is lambda* = sigma_p^2 / sigma_n^2 (noisier
  data -> smaller lambda). Same form as Schlemper et al.'s cascaded-CNN DC layer (IEEE TMI 2018) --
  verify the citation.

**Data and protocol.** fastMRI knee multicoil, CORPD_FBK, 15 coils (val + train_batch_0 pooled;
188 files), middle slice of each volume; 4x random Cartesian undersampling (fastMRI MaskFunc: 7%
fully sampled center, realized acceleration within +/-0.03 of 4), one mask per image seeded from
its k-space. Ground truth = ESPIRiT combination of the FULLY sampled k-space with the same maps --
NOT fastMRI's RSS target; state it. Splits: 30 tuning images (all settings), 100 evaluation images
(used once, everything fixed in advance), 4 reference images, all disjoint. Metrics PSNR, SSIM,
MS-SSIM, VIF, NMSE, HFEN on the central 320 x 320; paired per-image comparisons with Wilcoxon.
Settings chosen by grid search on the tuning images: composite score (PSNR 0.35, SSIM 0.35, VIF
0.15, HFEN 0.15 on per-image-centered, spread-normalized metrics), excluding settings clearly
worse on PSNR or SSIM, ties tested on paired differences (`analyze_grid_search.py`).

**Findings so far (full k-space, 4x; final numbers will be cropped-pipeline).**
- TV weight is the decisive setting; delta barely matters in 0.065-0.65 (< 0.15 dB) -- a
  robustness point. Huber beats plain MSE by ~2.3 dB (Run A) / ~1.3 dB (ensemble).
- Best settings: Run A delta 0.14, TV 1.2e-3; ensemble members gentler, delta 0.065, TV 4.75e-4
  (averaging already denoises).
- Run A, 100 images: PSNR 34.00 +/- 2.36 dB, SSIM 0.871; early stopping used ~56% of the budget.
- Ensemble vs Run A on the tuning images, each at its best settings: +0.42 dB, +0.003 SSIM, +0.026
  VIF, -0.021 HFEN -- a modest gain; the 100-image paired result is the number to report.
- Readout cropping: ~32% less time per fit for ~-0.15 dB PSNR and slightly lower VIF (8 images;
  100-image figure pending).
- Error patterns: residual aliasing at strong vertical edges (phase-encode direction) is the main
  reconstruction failure; low-PSNR images often have noisy references.

**Do not use / known issues.**
- Don't use grid_v2-v4, the max-normalized metric columns, the old 32.3/33.3 dB batch numbers, or
  knee-only metrics (Itai's decision).
- **Manual exclusions:** `KAVG_BATCH_MANUAL_EXCLUDE` (Section 2, since 2026-09-14) drops 6 files
  from the evaluation pool -- per its comment, "qualitatively problematic reconstructions".
  Excluding images because their reconstructions looked bad biases the evaluation; each needs a
  data-level reason (e.g. corrupt file, artifact in the fully sampled data) or must go back in.
  Resolve before the final evaluation runs.
- Two settings in the notebook have no effect and need not be described: Z_SOURCE options other
  than uniform (the MRI-VAE latent was found non-Gaussian and is unused), and the frequency
  regularization / Kaiser upsampler options (off).

**Still missing for the paper:** baselines (CS L1-wavelet, original ConvDecoder, DIP/Deep
Decoder, a pretrained fastMRI U-Net/VarNet for reference), ablations (DCSE, Huber+TV, early
stopping on/off, ensemble vs single, guided init, readout crop, soft vs hard DC), 8x, brain,
uncertainty-map calibration, ground-truth-free settings, low-field data. Related work to cite
(verify each): ConvDecoder (Darestani & Heckel, IEEE TCI 2021), Deep Image Prior (Ulyanov et al.),
Deep Decoder (Heckel & Hand), early stopping for DIP (arXiv:2112.06074), SSDU (Yaman et al. 2020),
ESPIRiT (Uecker et al. 2014), fastMRI (Zbontar et al.), Schlemper et al. 2018 (DC layer).

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
  overrides are silently ignored (this bug has happened more than once). Section 2's derived values
  live in `_derive_config()`, which runs again after the loop -- add new ones there.
- After changing the notebook, run the CPU end-to-end test (`tests/e2e_cpu/README.md`) before
  submitting long jobs: syntax checks don't catch a name defined only in a skip-slurm cell.
- Parallel jobs must write to distinct files (e.g. `results_shard<i>.csv`), then merge
  afterwards -- never share one output file between concurrent jobs.
- **Image selections are fixed and must not depend on hyperparameters**: the 100-image batch
  (`selections/kavg_batch_selection_n100.json`) and references (`kavg_ref_selection_kavg<K>.json`)
  are keyed only by size/K. Tuning images are never drawn from them. Change them only on purpose.
- Style: long explanatory comments that give the *why* next to each toggle, and detailed commit
  messages that say what changed, why, and which job ID exposed it. Match this.
- Markdown cells in the notebook are sometimes stale (e.g. "k=10", "N=20"); the code in Section 2
  is the source of truth.
