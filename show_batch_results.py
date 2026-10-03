#!/usr/bin/env python3
"""Look at a batch run's reconstructions (Run A or K-ensemble, Section 12.5). The batch saves only
the fitted members, not images, so this reloads a few of them -- no fitting -- rebuilds each
image's measurement exactly as the batch did (seeded mask), reconstructs, averages members like
the batch (uniform complex average, the current KAVG_* setting), and saves figures.

Which images: by default the best, two around the median, and the worst by PSNR in the run's
results CSV, so you see the range. --images picks them yourself.

Output (CKPT_ROOT/diagnostics/batch_examples/<results-csv stem>/):
  <image>.png     top: ground truth | zero-filled | reconstruction | |error| (0 to 0.15)
                  bottom: the same, zoomed into the yellow box; plus the anatomy mask
  overview.png    one row per image: ground truth | reconstruction | |error|
Each figure's title repeats the metrics recomputed here next to the CSV's, as a check.

Usage (via run_show_batch_results.sbatch):
    python3 show_batch_results.py                       # Run A (K=1, random init)
    python3 show_batch_results.py --k 4 --init guided   # K-ensemble
    python3 show_batch_results.py --images file1001234.h5,file1005678.h5
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=1, help="K_VALUE of the run (1 = Run A)")
    parser.add_argument("--init", default="random", choices=["random", "guided"], help="KAVG_INIT_MODE of the run")
    parser.add_argument("--images", help="comma-separated filenames (default: best/median/worst by PSNR)")
    parser.add_argument("--results-csv", help="the batch results CSV (default: found from the current config)")
    parser.add_argument("--out", help="output folder")
    args = parser.parse_args()
    os.environ["CD_K_VALUE"] = str(args.k)
    os.environ["CD_KAVG_INIT_MODE"] = args.init

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, repo_dir)
    import torch
    assert torch.cuda.is_available(), "No GPU -- run this through run_show_batch_results.sbatch."
    from view_grid_settings import load_notebook
    ns = load_notebook(repo_dir, skip_markers=("def _normality_report(",))

    import numpy as np
    import pandas as pd
    import h5py
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    crop = ns.get("maybe_readout_crop", lambda k: k)     # older notebooks have no READOUT_CROP
    crop_tag = ns.get("CROP_TAG", "")
    tag = (f"kavg{args.k}_{ns['ARCHITECTURE']}_{ns['Z_SOURCE']}{ns['ACCEL_LOSS_TAG']}{ns['BLUR_TAG']}"
           f"{ns['FREQ_REG_TAG']}{crop_tag}")
    member_dir = f"{ns['CKPT_SUBDIR']}/checkpoints_multicoil_{tag}" + ("_untrained" if args.init == "random" else "") + "_batch_acc"

    if args.results_csv:
        csv_path = args.results_csv
    else:
        stem = f"{ns['RESULTS_DIR']}/kavg_batch_k{args.k}_{ns['RUN_TAG']}" + ("_untrained" if args.init == "random" else "")
        cands = [p for p in glob.glob(stem + "_*.csv")
                 if not any(w in os.path.basename(p) for w in ("summary", "sweep", "fresh", "untrained_" if args.init == "guided" else "\0"))]
        assert len(cands) == 1, f"expected one results CSV for {stem}_*.csv, found {cands} -- pass --results-csv"
        csv_path = cands[0]
    df = pd.read_csv(csv_path)
    print(f"Results: {csv_path} ({len(df)} images)\nMember checkpoints: {member_dir}")

    if args.images:
        images = args.images.split(",")
    else:
        d = df.sort_values("PSNR").reset_index(drop=True)
        idx = sorted({len(d) - 1, len(d) // 2, max(len(d) // 2 - 1, 0), 0}, reverse=True)
        images = list(d.loc[idx, "Image"])
    refs = []
    if args.init == "guided":
        with open(f"{ns['SELECTIONS_DIR']}/kavg_ref_selection_kavg{args.k}.json") as f:
            refs = json.load(f)[:args.k]
    out_dir = args.out or os.path.join(ns["CKPT_ROOT"], "diagnostics", "batch_examples",
                                       os.path.splitext(os.path.basename(csv_path))[0])
    os.makedirs(out_dir, exist_ok=True)

    overview = []
    for fname in images:
        row = df[df["Image"] == fname]
        with h5py.File(os.path.join(ns["folder"], fname), "r") as f:
            slice_ksp = crop(f["kspace"][f["kspace"].shape[0] // 2])
        ksp_tt = torch.from_numpy(np.stack((slice_ksp.real, slice_ksp.imag), axis=-1))
        shape = {"output_depth": ksp_tt.shape[0] * 2, "out_size": tuple(ksp_tt.shape[1:-1])}
        net_for_scale = ns["build_network"](shape)
        ns["set_seed"](ns["SEED"])
        u = ns["build_undersampled"](ksp_tt, slice_ksp, net_for_scale)   # same steps as the batch
        stem = ns["stable_stem"](fname)
        paths = ([f"{member_dir}/target_{stem}__member_{i}.pt" for i in range(args.k)] if args.init == "random"
                 else [f"{member_dir}/target_{stem}__ref_{ns['stable_stem'](r)}.pt" for r in refs])
        recs = []
        for p in paths:
            assert os.path.exists(p), f"missing member checkpoint {p}"
            ckpt = torch.load(p, weights_only=False)
            assert np.array_equal(np.asarray(ckpt["mask1d"]), np.asarray(u["mask1d"])), (
                f"{p} was fit on a different mask than the one rebuilt here")
            net = ns["build_network"](shape, seed=ckpt.get("seed", ns["SEED"]))
            net.load_state_dict(ckpt["model_state_dict"])
            if ckpt.get("freq_frac") is not None:
                net.set_freq_frac(ckpt["freq_frac"])
            recs.append(ns["reconstruct"](net, ckpt["net_input"], u["mps"], masked_kspace=u["masked_kspace"],
                                          scaling_factor=ckpt["scaling_factor"], return_complex=True))
        w = np.full(len(recs), 1.0 / len(recs))
        if ns.get("KAVG_WEIGHTED_ENSEMBLE") and ns.get("KAVG_WEIGHT_EXPONENT", 0) != 0:
            print("  note: weighted averaging with a nonzero exponent isn't reproduced here -- uniform average used")
        rec = (np.abs(ns["kavg_weighted_average"](np.stack(recs), w)) if ns["KAVG_AVERAGE_COMPLEX"]
               else ns["kavg_weighted_average"](np.stack([np.abs(r) for r in recs]), w))
        gt, zf, r = ns["to_metric_scale"](u["gt_espirit"], u["zf_img_cropped"], rec)
        m = {**ns["compute_all_metrics"](gt, r), **ns["anatomy_metrics"](gt, r)}
        m_zf = ns["compute_all_metrics"](gt, zf)
        csv_note = (f" (CSV: {row['PSNR'].iloc[0]:.2f} dB / {row['SSIM'].iloc[0]:.3f})" if len(row) else "")
        print(f"{fname}: PSNR {m['PSNR']:.2f} dB, SSIM {m['SSIM']:.3f}, knee {m['PSNR_fg']:.2f} dB{csv_note}")

        rs, cs = ns["_display_bbox"](gt)
        H, W = gt[rs, cs].shape
        side = max(min(H, W) // 3, 32)
        r0, c0 = (H - side) // 2, (W - side) // 2
        fg = ns["foreground_mask"](gt)
        panels = [("Ground truth", gt), (f"Zero-filled\nPSNR {m_zf['PSNR']:.2f} | SSIM {m_zf['SSIM']:.3f}", zf),
                  (f"Reconstruction (K={args.k}, {args.init})\nPSNR {m['PSNR']:.2f} | SSIM {m['SSIM']:.3f}"
                   f"\nknee {m['PSNR_fg']:.2f} | {m['SSIM_fg']:.3f}", r)]
        fig, axes = plt.subplots(2, 4, figsize=(17, 9.5))
        for j, (title, img) in enumerate(panels):
            crop_img = img[rs, cs]
            axes[0, j].imshow(np.flipud(crop_img), cmap="gray", vmin=0, vmax=1)
            axes[0, j].set_title(title, fontsize=9)
            vmax = max(gt[rs, cs][r0:r0 + side, c0:c0 + side].max(), 1e-6)
            axes[1, j].imshow(np.flipud(crop_img[r0:r0 + side, c0:c0 + side]), cmap="gray", vmin=0, vmax=vmax)
            axes[1, j].set_title("zoom", fontsize=9)
        axes[0, 0].add_patch(Rectangle((c0, H - r0 - side), side, side, fill=False, edgecolor="yellow", lw=1))
        err = np.abs(gt - r)[rs, cs]
        axes[0, 3].imshow(np.flipud(err), cmap="gray", vmin=0, vmax=0.15)
        axes[0, 3].set_title("|error| of the reconstruction (0 to 0.15)", fontsize=9)
        axes[1, 3].imshow(np.flipud(fg[rs, cs]), cmap="gray", vmin=0, vmax=1)
        axes[1, 3].set_title("anatomy mask (knee-only metrics)", fontsize=9)
        for a in axes.ravel():
            a.axis("off")
        fig.suptitle(f"{fname}{csv_note}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=120)
        plt.close(fig)
        overview.append((fname, gt[rs, cs], r[rs, cs], err, m))

    fig, axes = plt.subplots(len(overview), 3, figsize=(11, 3.8 * len(overview)), squeeze=False)
    for i, (fname, g, r, e, m) in enumerate(overview):
        for j, (img, kw, title) in enumerate([(g, dict(vmin=0, vmax=1), f"{fname}: ground truth"),
                                              (r, dict(vmin=0, vmax=1), f"reconstruction, PSNR {m['PSNR']:.2f} | SSIM {m['SSIM']:.3f}"),
                                              (e, dict(vmin=0, vmax=0.15), "|error| (0 to 0.15)")]):
            axes[i, j].imshow(np.flipud(img), cmap="gray", **kw)
            axes[i, j].set_title(title, fontsize=8)
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overview.png"), dpi=110)
    plt.close(fig)
    print(f"\nSaved {len(overview)} figure(s) + overview.png in {out_dir}")


if __name__ == "__main__":
    main()
