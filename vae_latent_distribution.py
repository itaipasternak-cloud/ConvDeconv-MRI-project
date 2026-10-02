#!/usr/bin/env python3
"""Is the MRI-VAE latent normally distributed? A larger version of the notebook's Section 6.2
diagnostic, as a quick GPU job (run_vae_latent_check.sbatch, ~10 min).

For each image: the same zero-filled complex image a real fit would encode (seeded mask,
ESPIRiT combination, 320x320 crop -- build_undersampled()'s steps), passed through the frozen
microsoft/mri-autoencoder-v0.1 exactly as Z_SOURCE="mri_vae" does. The notebook's own code is
reused (Sections 2-6 are executed; nothing is re-implemented). Two things are analyzed:

  RAW latent:  posterior.mean, the VAE's 4-channel output (_mri_vae_raw_latent()).
  FINAL z:     what the network actually receives (mri_vae_encode_to_z(): channels repeated to
               256, resized to 8x4, min-max scaled to [0, 1]).

Normality is judged per latent channel as well as pooled: channels with different means or
spreads, pooled together, give a non-normal mixture even when every channel is normal on its own.
So the pooled raw latent is also reported after standardizing each channel. With this many
values, formal tests (Shapiro-Wilk on a 5000-value subsample, D'Agostino-Pearson) reject even
tiny deviations, so the effect sizes matter more: skewness and excess kurtosis (both 0 for a
normal) and the share of values beyond 3 standard deviations (0.27% for a normal).

Images: by default the 30 grid-search tuning images (grid_v2's manifest) -- encoding them uses
no ground truth and fits nothing, so it doesn't touch the evaluation set.

Output (CKPT_ROOT/diagnostics/vae_latent/ by default):
  vae_latent_overview.png     histogram + fitted normal, and Q-Q plot: raw (pooled), raw
                              (channels standardized), final z
  vae_latent_per_channel.png  the same for each of the 4 raw latent channels
  stats.csv                   every statistic printed at the end
  per_image.csv               per-image, per-channel mean/std/skew/kurtosis

Usage (via sbatch): python3 vae_latent_distribution.py [--n-images 30] [--images f1.h5,f2.h5]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CD_Z_SOURCE", "mri_vae")

SECTION_6_2_MARKER = "def _normality_report("   # the notebook's own 5-image version -- skipped
EXPECTED_BEYOND_3SD = 0.0027


def describe(values, label, rng):
    import numpy as np
    from scipy import stats
    v = np.asarray(values, dtype=np.float64).ravel()
    sub = v if v.size <= 5000 else rng.choice(v, 5000, replace=False)
    z = (v - v.mean()) / v.std()
    shapiro_w, shapiro_p = stats.shapiro(sub)
    return {
        "Scope": label, "N": v.size, "Mean": v.mean(), "Std": v.std(),
        "Skew": stats.skew(v), "Excess_Kurtosis": stats.kurtosis(v),
        "Beyond_3SD": float(np.mean(np.abs(z) > 3)),
        "Shapiro_W": shapiro_w, "Shapiro_p": shapiro_p,
        "DAgostino_p": stats.normaltest(v)[1],
    }


def hist_qq(ax_h, ax_q, values, title, color):
    import numpy as np
    from scipy import stats
    v = np.asarray(values, dtype=np.float64).ravel()
    ax_h.hist(v, bins=100, density=True, alpha=0.75, color=color)
    xs = np.linspace(v.min(), v.max(), 300)
    ax_h.plot(xs, stats.norm.pdf(xs, v.mean(), v.std()), "r--", lw=1.5, label="normal, same mean/std")
    ax_h.set_title(f"{title}\nskew {stats.skew(v):.2f}, excess kurtosis {stats.kurtosis(v):.2f}", fontsize=9)
    ax_h.legend(fontsize=7)
    stats.probplot(v if v.size <= 20000 else np.random.default_rng(0).choice(v, 20000, replace=False),
                   dist="norm", plot=ax_q)
    ax_q.set_title(f"{title}: normal Q-Q", fontsize=9)
    ax_q.get_lines()[0].set_markersize(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-images", type=int, default=30)
    parser.add_argument("--images", help="comma-separated filenames (default: grid_v2's tuning images)")
    parser.add_argument("--out", help="output folder (default CKPT_ROOT/diagnostics/vae_latent)")
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, repo_dir)
    import torch
    assert torch.cuda.is_available(), "No GPU -- run this through run_vae_latent_check.sbatch."
    from view_grid_settings import load_notebook
    ns = load_notebook(repo_dir, skip_markers=(SECTION_6_2_MARKER,))
    assert ns["Z_SOURCE"] == "mri_vae", "Z_SOURCE must be 'mri_vae' (CD_Z_SOURCE=mri_vae) so the VAE is loaded"

    import numpy as np
    import pandas as pd
    import h5py
    import matplotlib.pyplot as plt

    if args.images:
        images = args.images.split(",")
    else:
        manifest = os.path.join(ns["CKPT_ROOT"], "grid_search", "grid_v2", "manifest.json")
        if os.path.exists(manifest):
            with open(manifest) as f:
                images = json.load(f)["images"]
        else:
            images = list(ns["all_files"])
            np.random.default_rng(ns["SEED"]).shuffle(images)
    images = images[:args.n_images]
    out_dir = args.out or os.path.join(ns["CKPT_ROOT"], "diagnostics", "vae_latent")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Encoding {len(images)} images with {ns['MRI_VAE_MODEL_ID']} -> {out_dir}")

    raw_by_channel, final_all, per_image = None, [], []
    for n, fname in enumerate(images, 1):
        with h5py.File(os.path.join(ns["folder"], fname), "r") as f:
            slice_ksp = ns["maybe_readout_crop"](f["kspace"][f["kspace"].shape[0] // 2])
        ksp_tt = torch.from_numpy(np.stack((slice_ksp.real, slice_ksp.imag), axis=-1))
        try:
            # build_undersampled()'s steps up to the zero-filled image, without the network
            mask, _, _ = ns["get_mask"](ksp_tt, slice_ksp, factor=4, cent=0.07,
                                         seed=ns["mask_seed_for"](slice_ksp))
            masked_kspace, _ = ns["apply_mask"](ksp_tt, mask=mask)
            mps = ns["espirit_maps"](masked_kspace)
        except ns["EspiritCalibrationError"] as e:
            print(f"  [{n}/{len(images)}] {fname}: skipped (ESPIRiT: {e})")
            continue
        coils = ns["ifft2"](masked_kspace).cpu().numpy()
        zf = ns["crop_center"](ns["espirit_combine"](coils[..., 0] + 1j * coils[..., 1], mps), 320, 320)
        raw = ns["_mri_vae_raw_latent"](zf).detach().cpu().numpy()[0]       # (4, h, w)
        final = ns["mri_vae_encode_to_z"](zf).detach().cpu().numpy().ravel()
        if raw_by_channel is None:
            raw_by_channel = [[] for _ in range(raw.shape[0])]
        for c in range(raw.shape[0]):
            raw_by_channel[c].append(raw[c].ravel())
            v = raw[c].ravel()
            per_image.append({"Image": fname, "Channel": c, "Mean": v.mean(), "Std": v.std(),
                              "Skew": float(pd.Series(v).skew()), "Excess_Kurtosis": float(pd.Series(v).kurt())})
        final_all.append(final)
        print(f"  [{n}/{len(images)}] {fname}: raw latent {raw.shape}, final z {final.size} values")
    assert raw_by_channel, "no image could be encoded"

    rng = np.random.default_rng(0)
    channels = [np.concatenate(c) for c in raw_by_channel]
    pooled = np.concatenate(channels)
    standardized = np.concatenate([(c - c.mean()) / c.std() for c in channels])
    finals = np.concatenate(final_all)
    rows = [describe(pooled, "raw, pooled", rng), describe(standardized, "raw, channels standardized", rng)]
    rows += [describe(c, f"raw, channel {i}", rng) for i, c in enumerate(channels)]
    rows.append(describe(finals, "final z", rng))
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(out_dir, "stats.csv"), index=False)
    per_image_df = pd.DataFrame(per_image)
    per_image_df.to_csv(os.path.join(out_dir, "per_image.csv"), index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for j, (v, title, color) in enumerate([(pooled, "Raw latent, pooled", "steelblue"),
                                           (standardized, "Raw latent, each channel standardized", "seagreen"),
                                           (finals, "Final z (network input)", "darkorange")]):
        hist_qq(axes[0, j], axes[1, j], v, title, color)
    fig.suptitle(f"MRI-VAE latent distribution -- {len(final_all)} images", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "vae_latent_overview.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(channels), figsize=(4.2 * len(channels), 8.5))
    for c, v in enumerate(channels):
        hist_qq(axes[0, c], axes[1, c], v, f"Raw latent, channel {c}", "steelblue")
    fig.suptitle(f"MRI-VAE raw latent per channel -- {len(final_all)} images", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "vae_latent_per_channel.png"), dpi=130)
    plt.close(fig)

    pd.set_option("display.width", 220)
    fmt = lambda v: f"{v:.4g}"
    print("\nDistribution statistics (a normal distribution has skew 0, excess kurtosis 0, "
          f"{EXPECTED_BEYOND_3SD:.2%} of values beyond 3 SD):")
    print(stats_df.to_string(index=False, float_format=fmt))
    print("\nPer-image spread of the per-channel statistics (do images differ a lot?):")
    print(per_image_df.groupby("Channel")[["Mean", "Std", "Skew", "Excess_Kurtosis"]]
          .agg(["mean", "std", "min", "max"]).to_string(float_format=fmt))
    print(f"\nSaved vae_latent_overview.png, vae_latent_per_channel.png, stats.csv, per_image.csv in {out_dir}")


if __name__ == "__main__":
    main()
