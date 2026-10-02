# CPU end-to-end test of the notebook

Runs `MRI_ConvDeconv_variance_earlystop.ipynb` the way a SLURM job does -- cells tagged `skip-slurm` /
`no-unattended` stripped, config from `CD_*` environment variables -- on a CPU, with 14 small synthetic
k-space files (4 coils, 640x368 like real knee data) and the network shrunk to 32 channels. It checks that the code paths
run end to end (grid search, batch, resume, checkpoint reuse), not reconstruction quality. Each run
takes seconds to a minute. Use it before submitting a long cluster job after changing the notebook.

Needs a Python env with: torch torchvision sigpy h5py sewar pytorch_msssim scikit-image scipy pandas
matplotlib tqdm ipython nbconvert.

```bash
W=/tmp/cd_e2e && rm -rf $W && mkdir -p $W/data $W/results/selections $W/results/grid_search/grid_v2
python tests/e2e_cpu/make_data.py $W
COMMON="CD_NUM_COILS=0 CD_EARLY_STOP_PLOT=False CD_EARLY_STOP_PLOT_LIVE=False CD_num_iters_slow=6"
# accelerated grid search (fits the 4 references first)
python tests/e2e_cpu/run_nb.py $PWD $W/results $W/data $COMMON CD_RUN_GRID_SEARCH=True CD_GRID_REGIME=accel \
  CD_K_AVERAGING=False CD_RUN_BATCH_EVAL=False CD_ACCEL_NUM_ITERS=4 CD_GRID_NUM_IMAGES=2 \
  CD_GRID_HUBER_DELTA_VALUES='[0.14]' CD_GRID_TV_WEIGHT_VALUES='[1.2e-3, 3e-3]' CD_GRID_NUM_SHARDS=1
# Run A batch, 1 image (run twice to check resume)
python tests/e2e_cpu/run_nb.py $PWD $W/results $W/data $COMMON CD_RUN_GRID_SEARCH=False CD_K_AVERAGING=True \
  CD_RUN_BATCH_EVAL=True CD_K_VALUE=1 CD_KAVG_INIT_MODE=random CD_ACCEL_NUM_ITERS=5 \
  CD_USE_SEPARATE_ACCEL_PARAMS=False CD_KAVG_BATCH_SIZE=1
# K-ensemble batch, 1 image (reuses the references the grid search made)
python tests/e2e_cpu/run_nb.py $PWD $W/results $W/data $COMMON CD_RUN_GRID_SEARCH=False CD_K_AVERAGING=True \
  CD_RUN_BATCH_EVAL=True CD_ACCEL_NUM_ITERS=4 CD_KAVG_BATCH_SIZE=1
```

Each run ends with `E2E OK` or `!!! FAILED in cell <n>` plus the traceback. This test found the missing
`import pandas` that would have crashed every batch job once the skip-slurm stripping started working.
