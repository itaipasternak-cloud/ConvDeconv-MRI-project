#!/usr/bin/env python3
"""Refit a few grid-search tuning images under several (huber_delta, tv_weight) settings and save
side-by-side figures, to LOOK at what the grid search's numbers mean (e.g. whether a high
TV_WEIGHT over-smooths fine texture). The grid search itself never saves reconstructions, so
this has to refit -- run it on a GPU node via run_view_grid_settings.sbatch, not the login node.

Each fit is the grid search's own fit (Section 6.5): Run A regime, network seed SEED + 1000,
GRID num_iters from the search's manifest, the notebook's current early stopping / LR schedule,
hard data consistency, scored with compute_all_metrics(). The notebook's code is reused directly:
its setup cells (Sections 2-6, minus Section 1's git pull and the GUI/push cells) are executed
from MRI_ConvDeconv_variance_earlystop.ipynb, so nothing here re-implements the pipeline.

Two differences from the grid search, both on purpose:
  - Every setting of one image is fit on the SAME undersampling mask, seeded from the filename.
    (grid_v2-v4 ran before the notebook seeded its masks, so they compared settings on different
    masks; here the comparison is exact, and a preempted job resumes on the same masks. These
    filename-seeded masks differ from the notebook's own per-image masks, mask_seed_for().)
  - Besides full-frame PSNR/SSIM (what the grid search reports), PSNR/SSIM are also computed on
    the anatomy bounding box only ("ROI"), since strong TV also flattens background noise, which
    raises full-frame metrics without improving the knee itself.

Output (CKPT_ROOT/grid_search/viewer/ by default):
  <image>.png        rows: full image / |error| / zoomed patch; columns: GT, zero-filled, settings
  metrics.csv        full-frame and ROI metrics per (image, setting)
  cache/*.npz        one reconstruction per (image, setting) -- a requeued job skips these

Images: by default 3 of the grid's 30 tuning images, picked from the grid results as the image
where settings[0] beat settings[1] by the least, the median, and the most PSNR (so you see the
range, not one typical case). Pass --images to choose them yourself.

Usage (normally via sbatch -- see run_view_grid_settings.sbatch):
    python3 view_grid_settings.py
    python3 view_grid_settings.py --settings 0.3:1.2e-3,0.3:4.75e-4 --n-images 5
    python3 view_grid_settings.py --images file1000123.h5,file1000456.h5
"""
import argparse
import glob
import json
import os
import sys
import time
import zlib

os.environ.setdefault("MPLBACKEND", "Agg")

DEFAULT_SETTINGS = "0.3:1.2e-3,0.3:4.75e-4,0.65:1.9e-4,0.14:1.2e-5"
NOTEBOOK = "MRI_ConvDeconv_variance_earlystop.ipynb"
GRID_CELL_MARKER = "# Hyperparameter grid search for HUBER_DELTA / TV_WEIGHT"


def parse_settings(text):
    return [(float(d), float(tv)) for d, tv in (p.split(":") for p in text.split(","))]


def load_notebook(repo_dir):
    """Execute the notebook's setup cells (everything before Section 6.5's grid search) into one
    namespace, exactly as a SLURM run would, and return that namespace."""
    from IPython.core.inputtransformer2 import TransformerManager
    transform = TransformerManager().transform_cell
    with open(os.path.join(repo_dir, NOTEBOOK)) as f:
        nb = json.load(f)
    ns = {"__name__": "__notebook__", "REPO_DIR": repo_dir, "IN_COLAB": False}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        tags = cell.get("metadata", {}).get("tags", [])
        if GRID_CELL_MARKER in src:
            break
        if "skip-slurm" in tags or "no-unattended" in tags:
            continue
        if "REPO_SLUG" in src:  # Section 1: git pull / pip install / git identity -- not needed here
            continue
        exec(compile(transform(src), "<notebook cell>", "exec"), ns)
    return ns


def seed_masks(base_seed):
    """Make MaskFunc deterministic: each draw uses base_seed, base_seed+1, ... (get_mask() redraws
    until the acceleration is within tolerance, so the seed must advance between draws).
    Overrides the notebook's own per-image seed (mask_seed_for(), added after this script's first
    run, job 98026) so this script keeps its filename-based masks -- a requeued job then still
    matches the fits it already cached."""
    import demo_helper.helpers as H
    if not hasattr(H.MaskFunc, "_orig_call"):
        H.MaskFunc._orig_call = H.MaskFunc.__call__
        H.MaskFunc._next_seed = None

        def _seeded_call(self, shape, seed=None):
            if H.MaskFunc._next_seed is not None:
                seed = H.MaskFunc._next_seed
                H.MaskFunc._next_seed += 1
            return H.MaskFunc._orig_call(self, shape, seed)

        H.MaskFunc.__call__ = _seeded_call
    H.MaskFunc._next_seed = base_seed


def pick_images(grid_dirs, images, settings, n):
    """Images where settings[0] - settings[1] PSNR (from the grid results) is lowest, median,
    highest (n evenly spaced ranks)."""
    import numpy as np
    import pandas as pd
    paths = [p for d in grid_dirs for p in glob.glob(os.path.join(d, "results_shard*.csv"))]
    if n >= len(images) or len(settings) < 2 or not paths:
        return images[:n]
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

    def psnr_of(s):
        rows = df[np.isclose(df["Huber_Delta"], s[0]) & np.isclose(df["TV_Weight"], s[1])]
        return rows.drop_duplicates("Image").set_index("Image")["PSNR"]

    diff = (psnr_of(settings[0]) - psnr_of(settings[1])).dropna().sort_values()
    if len(diff) < n:
        return images[:n]
    picks = list(diff.index[np.linspace(0, len(diff) - 1, n).round().astype(int)])
    print(f"Picked by grid-search PSNR difference ({settings[0]} minus {settings[1]}):")
    for img in picks:
        print(f"  {img}: {diff[img]:+.2f} dB")
    return picks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", default=DEFAULT_SETTINGS,
                        help=f"comma-separated delta:tv pairs (default {DEFAULT_SETTINGS})")
    parser.add_argument("--grids", default="grid_v2,grid_v3,grid_v4",
                        help="grid searches (under CKPT_ROOT/grid_search) to take images/results from")
    parser.add_argument("--n-images", type=int, default=3)
    parser.add_argument("--images", help="comma-separated filenames (overrides the automatic pick)")
    parser.add_argument("--out", help="output folder (default CKPT_ROOT/grid_search/viewer)")
    args = parser.parse_args()
    settings = parse_settings(args.settings)

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, repo_dir)
    import torch
    assert torch.cuda.is_available(), "No GPU -- run this through run_view_grid_settings.sbatch."
    ns = load_notebook(repo_dir)

    import numpy as np
    import pandas as pd
    import h5py
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    grid_dirs = [os.path.join(ns["CKPT_ROOT"], "grid_search", g) for g in args.grids.split(",")]
    with open(os.path.join(grid_dirs[0], "manifest.json")) as f:
        manifest = json.load(f)
    cfg = manifest["config"]
    num_iters = cfg["num_iters"]
    current = {"architecture": ns["ARCHITECTURE"], "z_source": ns["Z_SOURCE"], "seed": ns["SEED"],
               "early_stop_method": ns["EARLY_STOP_METHOD"], "use_lr_schedule": ns["USE_LR_SCHEDULE"],
               "espirit_acs_width": ns["ESPIRIT_ACS_WIDTH"]}
    for k, v in current.items():
        if cfg.get(k) != v:
            print(f"WARNING: notebook {k}={v!r} but the grid search used {cfg.get(k)!r} -- "
                  f"these fits won't match the grid search's numbers exactly.")

    images = args.images.split(",") if args.images else pick_images(
        grid_dirs, manifest["images"], settings, args.n_images)
    out_dir = args.out or os.path.join(ns["CKPT_ROOT"], "grid_search", "viewer")
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"{len(images)} image(s) x {len(settings)} setting(s), {num_iters} iterations each -> {out_dir}")

    normalize, psnr, ssim = ns["normalize"], ns["psnr"], ns["ssim"]
    rows = []
    for fname in images:
        stem = ns["stable_stem"](fname)
        with h5py.File(os.path.join(ns["folder"], fname), "r") as f:
            slice_ksp = f["kspace"][f["kspace"].shape[0] // 2]
        ksp_tt = torch.from_numpy(np.stack((slice_ksp.real, slice_ksp.imag), axis=-1))
        shape = {"output_depth": ksp_tt.shape[0] * 2, "out_size": tuple(ksp_tt.shape[1:-1])}
        net_for_scale = ns["build_network"](shape)
        ns["set_seed"](ns["SEED"])
        seed_masks((zlib.crc32(stem.encode()) + ns["SEED"]) & 0x7FFFFFFF)
        u = ns["build_undersampled"](ksp_tt, slice_ksp, net_for_scale)
        gt = normalize(u["gt_espirit"].astype(np.float64))
        zf = normalize(u["zf_img_cropped"].astype(np.float64))
        rs, cs = ns["_display_bbox"](gt)

        recs = {}
        for d, tv in settings:
            cache = os.path.join(cache_dir, f"{stem}__delta{d:g}_tv{tv:g}.npz")
            if os.path.exists(cache):
                c = np.load(cache)
                recs[(d, tv)] = (c["rec"], int(c["iters"]))
                print(f"{fname} delta={d:g} tv={tv:g}: cached")
                continue
            t0 = time.time()
            try:
                net = ns["build_network"](shape, seed=ns["SEED"] + 1000)
                ni = ns["get_z"](u["zf_complex_cropped"])
                sf, ni = ns["get_scale_factor"](net, ns["num_channels"], ns["in_size"],
                                                u["masked_kspace"], u["mps"], ni=ni)
                meas = ns["Variable"]((u["masked_kspace"] * sf)[None, :]).type(ns["dtype"])
                _, net = ns["fit"](net=net, img_noisy_var=meas, num_channels=ns["num_channels"],
                                   net_input=ni, apply_f=ns["forwardm"], mask=u["mask2d"],
                                   mask1d=u["mask1d"], scaling_factor=sf, num_iter=num_iters,
                                   LR=0.01, checkpoint_dir=None, huber_delta=d, tv_weight=tv)
                rec = ns["reconstruct"](net, ni, u["mps"], masked_kspace=u["masked_kspace"],
                                        scaling_factor=sf, return_complex=True)
            except ns["ReconstructionDivergedError"] as e:
                print(f"{fname} delta={d:g} tv={tv:g}: DIVERGED ({e})")
                continue
            rec = normalize(np.abs(rec).astype(np.float64))
            iters = int(getattr(net, "iters_run", num_iters))
            np.savez_compressed(cache, rec=rec, iters=iters)
            recs[(d, tv)] = (rec, iters)
            print(f"{fname} delta={d:g} tv={tv:g}: done in {time.time() - t0:.0f}s ({iters} iter)")

        for (d, tv), (rec, iters) in recs.items():
            m = ns["compute_all_metrics"](gt, rec)
            m.update(PSNR_ROI=psnr(gt[rs, cs], rec[rs, cs]), SSIM_ROI=ssim(gt[rs, cs], rec[rs, cs]))
            m.update(Image=fname, Huber_Delta=d, TV_Weight=tv, Iters_Run=iters)
            rows.append(m)

        # --- figure: full / |error| / zoom, columns GT, zero-filled, then each setting ---
        panels = [("Ground truth", gt, None), ("Zero-filled", zf, None)] + [
            (f"delta={d:g}, TV={tv:g}", rec, (d, tv)) for (d, tv), (rec, _) in recs.items()]
        H, W = gt[rs, cs].shape
        side = max(min(H, W) // 3, 32)
        zr0, zc0 = (H - side) // 2, (W - side) // 2  # central patch of the anatomy
        fig, axes = plt.subplots(3, len(panels), figsize=(3.6 * len(panels), 11.5))
        for j, (title, img, key) in enumerate(panels):
            crop = img[rs, cs]
            axes[0, j].imshow(np.flipud(crop), cmap="gray", vmin=0, vmax=1)
            if key is not None:
                r = next(r for r in rows if r["Image"] == fname and (r["Huber_Delta"], r["TV_Weight"]) == key)
                title += (f"\nPSNR {r['PSNR']:.2f} | SSIM {r['SSIM']:.3f}"
                          f"\nROI {r['PSNR_ROI']:.2f} | {r['SSIM_ROI']:.3f}")
            elif j == 1:
                title += f"\nPSNR {psnr(gt, zf):.2f} | SSIM {ssim(gt, zf):.3f}"
            axes[0, j].set_title(title, fontsize=9)
            if j == 0:
                axes[0, j].add_patch(Rectangle((zc0, H - zr0 - side), side, side,
                                               fill=False, edgecolor="yellow", linewidth=1))
                axes[1, j].set_title("|error| (0 to 0.15)", fontsize=9)
            else:
                axes[1, j].imshow(np.flipud(np.abs(gt[rs, cs] - crop)), cmap="gray", vmin=0, vmax=0.15)
            patch = crop[zr0:zr0 + side, zc0:zc0 + side]
            axes[2, j].imshow(np.flipud(patch), cmap="gray", vmin=0, vmax=max(gt[rs, cs][zr0:zr0 + side, zc0:zc0 + side].max(), 1e-6))
            for i in range(3):
                axes[i, j].axis("off")
        axes[2, 0].set_title("zoom (yellow box)", fontsize=9)
        fig.suptitle(f"{fname} -- same mask for every setting, {num_iters} iter max", fontsize=11)
        fig.tight_layout()
        png = os.path.join(out_dir, f"{stem}.png")
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"Saved {png}")

    df = pd.DataFrame(rows)[["Image", "Huber_Delta", "TV_Weight", "PSNR", "PSNR_ROI", "SSIM",
                             "SSIM_ROI", "VIF", "HFEN", "NMSE", "MS-SSIM", "Iters_Run"]]
    df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nMean per setting:")
    print(df.groupby(["Huber_Delta", "TV_Weight"])[["PSNR", "PSNR_ROI", "SSIM", "SSIM_ROI", "VIF", "HFEN"]]
          .mean().to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
