#!/usr/bin/env python3
"""Soft vs hard data consistency, by re-scoring a finished batch run's saved fits -- no fitting.

reconstruct() applies HARD data consistency: at every acquired k-space location the network's
prediction is replaced by the measurement, which also puts the measurement noise back in. SOFT
data consistency blends the two at acquired locations:

    k = (lambda * k_measured + k_predicted) / (1 + lambda)     (unacquired locations: k_predicted)

lambda = 0 is no data consistency (the network's own output), lambda -> infinity is hard DC.
Data consistency is applied only when reconstructing, never during fitting, so every lambda can
be scored on the SAME fitted networks: for each image the network output is computed once and
each lambda is built from it (same FFT, per-coil correction, ESPIRiT combination and 320 x 320
crop as reconstruct(); the "hard" row reproduces the batch CSV -- printed as a check).

For an ensemble (--k 4 --init guided) each member is corrected with the same lambda before the
members are averaged, exactly as the batch does with hard DC.

METHODOLOGY: this re-scores an EVALUATION run, so it must not be used to CHOOSE lambda -- that would
tune on the test set. Choose lambda on the tuning images (a grid search with GRID_DC_LAMBDAS, ranked
jointly with delta/TV by analyze_grid_search.py), then use this script only to evaluate the chosen
lambda against hard DC on the evaluation set: --lambdas <chosen>,inf.

Output (CKPT_ROOT/diagnostics/soft_dc/<results-csv stem>/): results.csv (one row per image x
lambda) and, at the end of the log, mean metrics per lambda plus the paired difference to hard DC
(mean +/- standard error, Wilcoxon signed-rank p-value) over all images.

Usage (via run_soft_dc_test.sbatch):
    python3 soft_dc_test.py --full-kspace                 # Run A, full k-space (job 98645)
    python3 soft_dc_test.py                               # Run A on the notebook's current pipeline
    python3 soft_dc_test.py --k 4 --init guided           # K-ensemble
    python3 soft_dc_test.py --lambdas 0,0.1,1,10,inf --n-images 20
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
DEFAULT_LAMBDAS = "0,0.03,0.1,0.3,1,3,10,inf"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=1, help="K_VALUE of the run (1 = Run A)")
    parser.add_argument("--init", default="random", choices=["random", "guided"])
    parser.add_argument("--full-kspace", action="store_true",
                        help="the run was made without readout cropping (READOUT_CROP=False), e.g. job 98645")
    parser.add_argument("--lambdas", default=DEFAULT_LAMBDAS, help=f"comma-separated (default {DEFAULT_LAMBDAS})")
    parser.add_argument("--n-images", type=int, help="only the first N images of the run (default: all)")
    parser.add_argument("--results-csv", help="the batch results CSV (default: found from the config)")
    args = parser.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",")]
    os.environ["CD_K_VALUE"] = str(args.k)
    os.environ["CD_KAVG_INIT_MODE"] = args.init
    if args.full_kspace:
        os.environ["CD_READOUT_CROP"] = "False"

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, repo_dir)
    import torch
    assert torch.cuda.is_available(), "No GPU -- run this through run_soft_dc_test.sbatch."
    from view_grid_settings import load_notebook
    ns = load_notebook(repo_dir, skip_markers=("def _normality_report(",))

    import numpy as np
    import pandas as pd
    import h5py
    from scipy.stats import wilcoxon

    crop = ns.get("maybe_readout_crop", lambda k: k)
    tag = (f"kavg{args.k}_{ns['ARCHITECTURE']}_{ns['Z_SOURCE']}{ns['ACCEL_LOSS_TAG']}{ns['BLUR_TAG']}"
           f"{ns['FREQ_REG_TAG']}{ns.get('CROP_TAG', '')}")
    member_dir = f"{ns['CKPT_SUBDIR']}/checkpoints_multicoil_{tag}" + ("_untrained" if args.init == "random" else "") + "_batch_acc"
    if args.results_csv:
        csv_path = args.results_csv
    else:
        stem = f"{ns['RESULTS_DIR']}/kavg_batch_k{args.k}_{ns['RUN_TAG']}" + ("_untrained" if args.init == "random" else "")
        cands = [p for p in glob.glob(stem + "_*.csv")
                 if not any(w in os.path.basename(p) for w in ("summary", "sweep", "fresh", "untrained_" if args.init == "guided" else "\0"))]
        assert len(cands) == 1, f"expected one results CSV for {stem}_*.csv, found {cands} -- pass --results-csv"
        csv_path = cands[0]
    batch = pd.read_csv(csv_path)
    images = list(batch["Image"])[:args.n_images] if args.n_images else list(batch["Image"])
    refs = []
    if args.init == "guided":
        with open(f"{ns['SELECTIONS_DIR']}/kavg_ref_selection_kavg{args.k}.json") as f:
            refs = json.load(f)[:args.k]
    out_dir = os.path.join(ns["CKPT_ROOT"], "diagnostics", "soft_dc", os.path.splitext(os.path.basename(csv_path))[0])
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "results.csv")
    done = set(pd.read_csv(out_csv)["Image"]) if os.path.exists(out_csv) else set()
    print(f"Run: {csv_path} ({len(images)} images, {len(done)} already re-scored)\n"
          f"Members: {member_dir}\nlambdas: {lambdas}")

    fft2, ifft2, dtype = ns["fft2"], ns["ifft2"], ns["dtype"]

    def corrected_coils(out_coils, masked_kspace, sf, lam):
        """Per-coil images after data consistency with weight lam (inf = hard, 0 = none), on the
        raw scale -- apply_data_consistency()'s steps with the hard replacement generalized."""
        out_ri = torch.from_numpy(np.stack((out_coils.real, out_coils.imag), axis=-1)).type(masked_kspace.dtype)
        pred = fft2(out_ri).to(masked_kspace.device)
        meas = masked_kspace * sf
        acquired = torch.any(masked_kspace != 0, dim=-1, keepdim=True).expand_as(masked_kspace)
        if np.isinf(lam):
            k = torch.where(acquired, meas, pred)
        else:
            k = torch.where(acquired, (lam * meas + pred) / (1.0 + lam), pred)
        imgs = ifft2(k / sf).cpu().numpy()
        return imgs[..., 0] + 1j * imgs[..., 1]

    for n, fname in enumerate(images, 1):
        if fname in done:
            continue
        with h5py.File(os.path.join(ns["folder"], fname), "r") as f:
            slice_ksp = crop(f["kspace"][f["kspace"].shape[0] // 2])
        ksp_tt = torch.from_numpy(np.stack((slice_ksp.real, slice_ksp.imag), axis=-1))
        shape = {"output_depth": ksp_tt.shape[0] * 2, "out_size": tuple(ksp_tt.shape[1:-1])}
        net_for_scale = ns["build_network"](shape)
        ns["set_seed"](ns["SEED"])
        u = ns["build_undersampled"](ksp_tt, slice_ksp, net_for_scale)
        stem = ns["stable_stem"](fname)
        paths = ([f"{member_dir}/target_{stem}__member_{i}.pt" for i in range(args.k)] if args.init == "random"
                 else [f"{member_dir}/target_{stem}__ref_{ns['stable_stem'](r)}.pt" for r in refs])
        members = []                                       # (network output per coil, scaling factor)
        for p in paths:
            ckpt = torch.load(p, weights_only=False)
            assert np.array_equal(np.asarray(ckpt["mask1d"]), np.asarray(u["mask1d"])), f"{p}: different mask"
            net = ns["build_network"](shape, seed=ckpt.get("seed", ns["SEED"]))
            net.load_state_dict(ckpt["model_state_dict"])
            if ckpt.get("freq_frac") is not None:
                net.set_freq_frac(ckpt["freq_frac"])
            with torch.no_grad():
                out = net(ckpt["net_input"].type(dtype)).data.cpu().numpy()[0]
            members.append((ns["channels2imgs_complex"](out), ckpt["scaling_factor"]))
        rows = []
        for lam in lambdas:
            recs = [ns["crop_center"](ns["espirit_combine"](corrected_coils(c, u["masked_kspace"], sf, lam), u["mps"]), 320, 320)
                    for c, sf in members]
            w = np.full(len(recs), 1.0 / len(recs))
            rec = (np.abs(ns["kavg_weighted_average"](np.stack(recs), w)) if ns["KAVG_AVERAGE_COMPLEX"]
                   else ns["kavg_weighted_average"](np.stack([np.abs(r) for r in recs]), w))
            g, r = ns["to_metric_scale"](u["gt_espirit"], rec)
            rows.append({"Image": fname, "Lambda": lam, **ns["compute_all_metrics"](g, r)})
        hard = next(r for r in rows if np.isinf(r["Lambda"]))
        csv_psnr = batch.loc[batch["Image"] == fname, "PSNR"].iloc[0]
        best = max(rows, key=lambda r: r["PSNR"])
        print(f"[{n}/{len(images)}] {fname}: hard {hard['PSNR']:.3f} dB (batch CSV {csv_psnr:.3f}), "
              f"no DC {rows[0]['PSNR']:.3f}, best lambda={best['Lambda']:g} {best['PSNR']:.3f}")
        pd.DataFrame(rows).to_csv(out_csv, mode="a", index=False, header=not os.path.exists(out_csv))

    df = pd.read_csv(out_csv)
    df = df[df["Image"].isin(images)]
    metrics = ["PSNR", "SSIM", "MS-SSIM", "VIF", "NMSE", "HFEN"]
    pd.set_option("display.width", 200)
    print(f"\nMean per lambda over {df['Image'].nunique()} images (inf = hard DC, the current setting; 0 = no DC):")
    print(df.groupby("Lambda")[metrics].mean().to_string(float_format=lambda v: f"{v:.4g}"))
    hard = df[np.isinf(df["Lambda"])].set_index("Image")
    print("\nPaired difference to hard DC (lambda minus hard), mean +/- standard error, Wilcoxon p:")
    for lam in sorted(df["Lambda"].unique()):
        if np.isinf(lam):
            continue
        cur = df[df["Lambda"] == lam].set_index("Image").loc[hard.index]
        parts = []
        for m in ["PSNR", "SSIM", "VIF", "HFEN"]:
            d = (cur[m] - hard[m]).values
            p = wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else float("nan")
            parts.append(f"{m} {d.mean():+.4g} +/- {d.std(ddof=1) / np.sqrt(len(d)):.2g} (p={p:.2g})")
        print(f"  lambda={lam:g}: " + ", ".join(parts))
    print(f"\nResults: {out_csv}")


if __name__ == "__main__":
    main()
