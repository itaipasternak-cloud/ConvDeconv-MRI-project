#!/usr/bin/env python3
"""Should ConvDecoder fit readout-cropped k-space instead of the full, 2x-oversampled k-space?

fastMRI knee k-space is 640 (readout) x 368 (phase-encode). The readout direction is fully
sampled and 2x oversampled: the 640 rows cover twice the field of view that is evaluated (the
central 320 x 320). Because every readout sample is measured, the oversampling can be removed
exactly -- inverse FFT along readout, keep the central half of the rows, FFT back -- giving
320 x 368 k-space with the same measured phase-encode columns. The network then fits only the
evaluated field of view, at roughly half the cost per iteration. (Phase-encode, the undersampled
direction, cannot be cropped like this.)

This compares, per image and per TV weight, a Run A-regime fit (random init, seed SEED + 1000,
6000 iterations with the notebook's early stopping/LR schedule -- exactly like the grid search) on
  "full":    the k-space as stored (640 x 368)
  "cropped": the readout-cropped k-space (320 x 368)
with the SAME undersampling mask (the full slice's seed is used for both) and the same scoring:
reconstruct() with hard data consistency, 320 x 320 crop, raw-scale metrics + anatomy-only
PSNR/SSIM. TV is varied because it is averaged over the network output, which shrinks by half --
the best TV weight may shift. It also records iterations run and seconds per iteration.

Sanity check printed per image: the two pipelines' ground truths (each from its own k-space and
ESPIRiT maps) should agree closely; a large difference would mean the crop is misaligned.

Output (CKPT_ROOT/diagnostics/readout_crop_test/): results.csv (one row per image x TV x mode,
appended after every fit -- a requeued job resumes), and a summary at the end of the log.

Usage (via run_readout_crop_test.sbatch):
    python3 readout_crop_test.py [--n-images 8] [--tv 4.75e-4,1.2e-3,3e-3] [--delta 0.14]
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")


def readout_crop(slice_ksp):
    """(coils, rows, cols) complex k-space, rows = readout. Inverse FFT along readout, keep the
    central half of the rows (the evaluated field of view), FFT back. Centered (fftshift)
    convention, as fastMRI stores k-space; norm='ortho' both ways."""
    import numpy as np
    rows = slice_ksp.shape[1]
    img = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(slice_ksp, axes=1), axis=1, norm="ortho"), axes=1)
    keep = img[:, rows // 4: rows // 4 + rows // 2, :]
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(keep, axes=1), axis=1, norm="ortho"),
                           axes=1).astype(slice_ksp.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument("--tv", default="4.75e-4,1.2e-3,3e-3", help="comma-separated TV weights")
    parser.add_argument("--delta", type=float, default=0.14)
    parser.add_argument("--num-iters", type=int, default=6000)
    parser.add_argument("--out", help="output folder (default CKPT_ROOT/diagnostics/readout_crop_test)")
    args = parser.parse_args()
    tvs = [float(t) for t in args.tv.split(",")]

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, repo_dir)
    import torch
    assert torch.cuda.is_available(), "No GPU -- run this through run_readout_crop_test.sbatch."
    from view_grid_settings import load_notebook
    ns = load_notebook(repo_dir, skip_markers=("def _normality_report(",))

    import numpy as np
    import pandas as pd
    import h5py

    with open(os.path.join(ns["CKPT_ROOT"], "grid_search", "grid_v2", "manifest.json")) as f:
        images = json.load(f)["images"][:args.n_images]
    out_dir = args.out or os.path.join(ns["CKPT_ROOT"], "diagnostics", "readout_crop_test")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    done = set()
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        done = set(zip(prev["Image"], prev["Mode"], prev["TV_Weight"].round(12)))
    print(f"{len(images)} images x {len(tvs)} TV weights x 2 modes, delta={args.delta:g}, "
          f"{args.num_iters} iterations max -> {csv_path} ({len(done)} fits already done)")

    real_mask_seed_for = ns["mask_seed_for"]
    for n, fname in enumerate(images, 1):
        with h5py.File(os.path.join(ns["folder"], fname), "r") as f:
            full = f["kspace"][f["kspace"].shape[0] // 2]
        seed = real_mask_seed_for(full)
        ns["mask_seed_for"] = lambda _ksp, _s=seed: _s   # same mask for both modes
        prepared, gts = {}, {}
        for mode, ksp in (("full", full), ("cropped", readout_crop(full))):
            if all((fname, mode, round(tv, 12)) in done for tv in tvs):
                continue
            ksp_tt = torch.from_numpy(np.stack((ksp.real, ksp.imag), axis=-1))
            shape = {"output_depth": ksp_tt.shape[0] * 2, "out_size": tuple(ksp_tt.shape[1:-1])}
            ns["set_seed"](ns["SEED"])
            u = ns["build_undersampled"](ksp_tt, ksp, ns["build_network"](shape))
            prepared[mode] = (ksp_tt, shape, u)
            gts[mode] = ns["to_metric_scale"](u["gt_espirit"])[0]
        if len(gts) == 2:
            rel = np.linalg.norm(gts["full"] - gts["cropped"]) / np.linalg.norm(gts["full"])
            print(f"[{n}/{len(images)}] {fname}: ground truths full vs cropped differ by {rel:.2%} "
                  f"(relative L2; small = crop aligned)")
        for mode, (ksp_tt, shape, u) in prepared.items():
            for tv in tvs:
                if (fname, mode, round(tv, 12)) in done:
                    continue
                net = ns["build_network"](shape, seed=ns["SEED"] + 1000)
                ni = ns["get_z"](u["zf_complex_cropped"])
                sf, ni = ns["get_scale_factor"](net, ns["num_channels"], ns["in_size"],
                                                u["masked_kspace"], u["mps"], ni=ni)
                meas = ns["Variable"]((u["masked_kspace"] * sf)[None, :]).type(ns["dtype"])
                torch.cuda.synchronize()
                t0 = time.time()
                _, net = ns["fit"](net=net, img_noisy_var=meas, num_channels=ns["num_channels"],
                                   net_input=ni, apply_f=ns["forwardm"], mask=u["mask2d"],
                                   mask1d=u["mask1d"], scaling_factor=sf, num_iter=args.num_iters,
                                   LR=0.01, checkpoint_dir=None, huber_delta=args.delta, tv_weight=tv)
                torch.cuda.synchronize()
                secs = time.time() - t0
                rec = np.abs(ns["reconstruct"](net, ni, u["mps"], masked_kspace=u["masked_kspace"],
                                               scaling_factor=sf, return_complex=True))
                g, r = ns["to_metric_scale"](u["gt_espirit"], rec)
                row = {"Image": fname, "Mode": mode, "Huber_Delta": args.delta, "TV_Weight": tv,
                       "Kspace_Shape": "x".join(map(str, shape["out_size"])),
                       **ns["compute_all_metrics"](g, r), **ns["anatomy_metrics"](g, r),
                       "Iters_Run": net.iters_run, "Fit_Seconds": round(secs, 1),
                       "Sec_Per_Iter": secs / net.iters_run}
                pd.DataFrame([row]).to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))
                print(f"  {fname} {mode:>7} tv={tv:g}: PSNR {row['PSNR']:.2f}  SSIM {row['SSIM']:.4f}  "
                      f"knee PSNR {row['PSNR_fg']:.2f}  ({net.iters_run} iter, {secs:.0f}s, "
                      f"{1000 * row['Sec_Per_Iter']:.1f} ms/iter)")
        ns["mask_seed_for"] = real_mask_seed_for

    df = pd.read_csv(csv_path)
    df = df[df["Image"].isin(images)]
    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.4g}"
    cols = ["PSNR", "SSIM", "VIF", "HFEN", "PSNR_fg", "SSIM_fg", "Iters_Run", "Fit_Seconds", "Sec_Per_Iter"]
    print("\nMean per mode and TV weight:")
    print(df.groupby(["TV_Weight", "Mode"])[cols].mean().to_string(float_format=fmt))
    print("\nPaired difference, cropped minus full (same image, same TV), mean +/- standard error:")
    for tv in sorted(df["TV_Weight"].unique()):
        p = df[df["TV_Weight"] == tv].pivot(index="Image", columns="Mode", values=["PSNR", "SSIM", "PSNR_fg", "Fit_Seconds"])
        p = p.dropna()
        if p.empty:
            continue
        parts = []
        for m in ["PSNR", "SSIM", "PSNR_fg", "Fit_Seconds"]:
            d = p[(m, "cropped")] - p[(m, "full")]
            parts.append(f"{m} {d.mean():+.4g} +/- {d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan'):.2g}")
        print(f"  tv={tv:g} (n={len(p)}): " + ", ".join(parts))
    speed = df.groupby("Mode")["Sec_Per_Iter"].mean()
    if {"full", "cropped"} <= set(speed.index):
        print(f"\nTime per iteration: cropped / full = {speed['cropped'] / speed['full']:.2f}")
    print(f"\nResults: {csv_path}")


if __name__ == "__main__":
    main()
