#!/usr/bin/env python3
"""Rank the (huber_delta, tv_weight) combinations of a finished grid search (notebook Section 6.5,
run_knee_grid_search.sbatch) and pick the winner.

Reads CKPT_ROOT/grid_search/<name>/manifest.json + results_shard*.csv (one row per image x
combination, all six metrics). Runs on the cluster login node -- no GPU, just pandas.

How the combined score works
----------------------------
Every combination was fit on the same images, so each image's own difficulty is removed first:
for every image and metric, subtract that image's mean over all combinations. What's left is how
much better/worse a combination did than average ON THAT IMAGE. Each metric is then divided by
its typical spread between combinations (pooled std over all images), so all metrics are in the
same units ("typical between-setting differences") whatever their native scale. The score is the
weighted sum, averaged over images:

    score = 0.35*PSNR + 0.35*SSIM + 0.15*VIF + 0.15*(-HFEN)     (default --weights)

PSNR and SSIM carry most of the weight; VIF and HFEN guard against over-smoothing (lost fine
texture). MS-SSIM and NMSE are reported but not scored (redundant with SSIM and PSNR). The score
depends only on this search's own results -- no reference to an earlier "known good" setting.

Guard: the winner must not be clearly worse (paired, ~95%) than the best combination on PSNR or
on SSIM, so a VIF/HFEN gain can't buy a real PSNR/SSIM loss. Combinations that diverged on any
image are never eligible.

Several searches run on the SAME images with the SAME config (e.g. a follow-up grid made with
GRID_REUSE_FROM) can be ranked together as one table: pass --grid-dir once per search. Settings
two searches share are counted once. Results are then saved to the LAST --grid-dir as
summary_combined.csv / best_combined.json / grid_heatmaps_combined.png.

Intensity scale (--scale): the standard metric columns compare each image divided by its OWN
max, so one bright artifact pixel in a reconstruction darkens all of it and is scored as error
everywhere (view_grid_settings.py found a fit scoring 22.3 dB that way vs 31.2 dB on the raw
scale). grid_v5 recorded raw-scale (*_raw) and least-squares (*_ls) columns next to the standard
max-normalized ones. Since then the notebook computes every metric on the raw scale
(to_metric_scale()), so the standard columns ARE raw -- manifest config "metric_scale": "raw".
Anatomy-only PSNR_fg/SSIM_fg are reported alongside when present.
--scale auto (default) ranks on raw whenever the results have it, otherwise on max.

Usage:
    python3 analyze_grid_search.py --grid-dir ~/fastmri_results/knee/grid_search/grid_v5 --scale raw --plot
    python3 analyze_grid_search.py --grid-dir ~/fastmri_results/knee/grid_search/grid_v2
    python3 analyze_grid_search.py --grid-dir ... --weights PSNR=0.5,SSIM=0.5 --plot
    python3 analyze_grid_search.py --grid-dir .../grid_v2 --grid-dir .../grid_v3 --grid-dir .../grid_v4 --plot
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

METRICS = ["PSNR", "SSIM", "MS-SSIM", "VIF", "NMSE", "HFEN"]
HIGHER_IS_BETTER = {"PSNR": True, "SSIM": True, "MS-SSIM": True, "VIF": True,
                    "NMSE": False, "HFEN": False}
DEFAULT_WEIGHTS = "PSNR=0.35,SSIM=0.35,VIF=0.15,HFEN=0.15"
Z = 2.0  # ~95% two-sided for "clearly worse" / "tied" checks


def parse_weights(text):
    weights = {}
    for part in text.split(","):
        name, value = part.split("=")
        name = name.strip()
        assert name in METRICS, f"unknown metric {name!r} in --weights (choose from {METRICS})"
        weights[name] = float(value)
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def paired_diff(per_image, a, b):
    """Mean and standard error of (combo a - combo b) over images where both have a value."""
    d = (per_image[a] - per_image[b]).dropna()
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-dir", required=True, action="append",
                        help="CKPT_ROOT/grid_search/<GRID_SEARCH_NAME>; repeat to rank several searches together")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help=f"metric weights for the combined score (default {DEFAULT_WEIGHTS})")
    parser.add_argument("--allow-partial", action="store_true",
                        help="analyze an unfinished search, using only images every combination has finished")
    parser.add_argument("--plot", action="store_true", help="save heatmaps (needs matplotlib)")
    parser.add_argument("--scale", choices=["auto", "max", "raw", "ls"], default="auto",
                        help="which intensity-scale convention's metrics to rank on (see above)")
    args = parser.parse_args()

    grid_dirs = [os.path.expanduser(d) for d in args.grid_dir]
    out_dir = grid_dirs[-1]
    suffix = "_combined" if len(grid_dirs) > 1 else ""
    weights = parse_weights(args.weights)

    # Settings are identified by their (delta, tv) VALUES, not each search's own combo numbering,
    # so several searches merge into one table.
    manifests, frames, pairs = [], [], set()
    for d in grid_dirs:
        with open(os.path.join(d, "manifest.json")) as f:
            m = json.load(f)
        manifests.append(m)
        pairs |= {(float(x), float(y)) for x in m["huber_delta_values"] for y in m["tv_weight_values"]}
        shard_files = sorted(glob.glob(os.path.join(d, "results_shard*.csv")))
        assert shard_files, f"no results_shard*.csv in {d}"
        frames.append(pd.concat([pd.read_csv(p) for p in shard_files], ignore_index=True))
    for d, m in zip(grid_dirs[1:], manifests[1:]):
        assert m["images"] == manifests[0]["images"] and m["config"] == manifests[0]["config"], (
            f"{d} used different images or config than {grid_dirs[0]} -- they can't be ranked together.")
    combos = sorted(pairs)
    combo_of = {p: c for c, p in enumerate(combos)}
    deltas = sorted({d for d, _ in combos})
    tvs = sorted({tv for _, tv in combos})
    n_images = len(manifests[0]["images"])
    if len(grid_dirs) > 1:
        print(f"Ranking {len(grid_dirs)} searches together ({', '.join(os.path.basename(d) for d in grid_dirs)}): "
              f"{len(combos)} distinct settings on the same {n_images} images.")

    df = pd.concat(frames, ignore_index=True)

    # What the standard PSNR/SSIM/... columns hold: "raw" for searches made after the notebook
    # switched to raw-scale metrics, "max" (no key) for grid_v5 and earlier.
    bases = {m["config"].get("metric_scale", "max") for m in manifests}
    assert len(bases) == 1, f"these searches' metric columns are on different scales ({bases}) -- rank them separately"
    base = bases.pop()

    def has_scale(scale):
        return scale == base or all(f"{m}_{scale}" in df.columns and df[f"{m}_{scale}"].notna().any()
                                    for m in METRICS)
    scale = args.scale if args.scale != "auto" else ("raw" if has_scale("raw") else "max")
    if not has_scale(scale):
        raise SystemExit(f"--scale {scale}: these results have no {scale}-scale metrics (their standard "
                         f"columns are {base}-scale). Use --scale {base}.")
    if scale != base:
        for m in METRICS:
            df[m] = df[f"{m}_{scale}"]
    print({"max": "Metrics: standard (each image divided by its own max).",
           "raw": "Metrics: raw scale (reconstruction not rescaled; both divided by the ground truth's max).",
           "ls": "Metrics: least-squares brightness match."}[scale])
    # Anatomy-only PSNR/SSIM (raw scale) are reported alongside, not scored.
    extra = [c for c in ("PSNR_fg", "SSIM_fg") if c in df.columns and df[c].notna().any()]
    if scale != "max":
        suffix += f"_{scale}"
    df["Combo"] = [combo_of[(float(d), float(tv))] for d, tv in zip(df["Huber_Delta"], df["TV_Weight"])]
    df = df.drop_duplicates(subset=["Image", "Combo"], keep="first")

    # --- completeness: only images that every combination has finished are comparable ---
    per_image_count = df.groupby("Image")["Combo"].nunique()
    complete_images = sorted(per_image_count[per_image_count == len(combos)].index)
    print(f"{len(df)}/{n_images * len(combos)} fits recorded; {len(complete_images)}/{n_images} "
          f"images finished for all {len(combos)} combinations.")
    if len(complete_images) < n_images:
        if not args.allow_partial:
            raise SystemExit("Search not finished -- wait for all shards, or pass --allow-partial "
                             "to look at the images finished so far.")
        print("  --allow-partial: analyzing the finished images only.")
    assert len(complete_images) >= 3, "need at least 3 fully finished images"
    df = df[df["Image"].isin(complete_images)]

    diverged = df[df["Status"] != "ok"].groupby("Combo").size()
    if len(diverged):
        print("Diverged fits (these combinations are not eligible):")
        for c, n in diverged.items():
            print(f"  delta={combos[c][0]:g} tv={combos[c][1]:g}: {n} image(s)")

    # --- per-metric image x combo tables ---
    tables = {m: df.pivot(index="Image", columns="Combo", values=m) for m in METRICS + extra}

    # --- combined score: remove each image's difficulty, then put metrics on a common scale ---
    # Dividing by each metric's own spread is what makes the weights scale-free: SSIM moves by
    # hundredths between settings while PSNR moves by tenths of a dB, so raw weights would let
    # PSNR swamp SSIM. After this, one unit of any metric = its typical between-setting change.
    score = 0
    spreads = {}
    for m, w in weights.items():
        centered = tables[m].sub(tables[m].mean(axis=1), axis=0)
        spread = np.nanstd(centered.values, ddof=1)
        spreads[m] = spread
        z = centered / spread if spread > 0 else centered * 0
        score = score + w * (z if HIGHER_IS_BETTER[m] else -z)
    tables["Score"] = score
    print("\nCommon scale -- one score unit equals this much of each metric (its typical change "
          "between settings on the same image):")
    for m, spread in spreads.items():
        print(f"  {m:<5} {spread:.4g}   (weight {weights[m]:.2f})")

    rows = []
    for c, (d, tv) in enumerate(combos):
        r = {"Combo": c, "Huber_Delta": d, "TV_Weight": tv,
             "Diverged": int(diverged.get(c, 0))}
        for m in METRICS + extra + ["Score"]:
            col = tables[m][c].dropna()
            r[m] = col.mean()
            r[f"{m}_SE"] = col.std(ddof=1) / np.sqrt(len(col))
        rows.append(r)
    summary = pd.DataFrame(rows)

    # --- guard: not clearly worse than the best combination on PSNR or SSIM ---
    best_psnr = int(summary["PSNR"].idxmax())
    best_ssim = int(summary["SSIM"].idxmax())
    for m, best in (("PSNR", best_psnr), ("SSIM", best_ssim)):
        ok = []
        for c in summary["Combo"]:
            mean, se = paired_diff(tables[m], c, best) if c != best else (0.0, 0.0)
            ok.append(mean + Z * se >= 0)
        summary[f"OK_vs_best_{m}"] = ok
    summary["Eligible"] = (summary["Diverged"] == 0) & summary["OK_vs_best_PSNR"] & summary["OK_vs_best_SSIM"]

    ranked = summary.sort_values("Score", ascending=False)
    eligible = ranked[ranked["Eligible"]]
    if eligible.empty:
        print("\nWARNING: no combination passes the PSNR/SSIM guard -- falling back to the top score.")
        eligible = ranked[ranked["Diverged"] == 0]
    winner = eligible.iloc[0]
    w_combo = int(winner["Combo"])

    # --- report ---
    pd.set_option("display.width", 200)
    show = ["Huber_Delta", "TV_Weight", "Score", "PSNR", "PSNR_SE", "SSIM", "VIF", "HFEN",
            "MS-SSIM", "NMSE"] + extra + ["Eligible"]
    print(f"\nWeights: {', '.join(f'{k}={v:.2f}' for k, v in weights.items())}   "
          f"(n={len(complete_images)} images, paired)")
    print(ranked[show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print(f"\nBest mean PSNR: delta={combos[best_psnr][0]:g} tv={combos[best_psnr][1]:g} "
          f"({summary.loc[best_psnr, 'PSNR']:.3f} dB)")
    print(f"Best mean SSIM: delta={combos[best_ssim][0]:g} tv={combos[best_ssim][1]:g} "
          f"({summary.loc[best_ssim, 'SSIM']:.4f})")

    tied = []
    for c in summary["Combo"]:
        if c == w_combo or summary.loc[c, "Diverged"]:
            continue
        mean, se = paired_diff(tables["Score"], w_combo, c)
        if mean - Z * se < 0:
            tied.append(c)
    print(f"\nWINNER: HUBER_DELTA = {winner['Huber_Delta']:g}, TV_WEIGHT = {winner['TV_Weight']:g}")
    print(f"  PSNR {winner['PSNR']:.3f} dB, SSIM {winner['SSIM']:.4f}, VIF {winner['VIF']:.4f}, "
          f"HFEN {winner['HFEN']:.4g}")
    if tied:
        print(f"  Statistically tied with {len(tied)} other combination(s) on the score: "
              + ", ".join(f"({combos[c][0]:g}, {combos[c][1]:g})" for c in tied)
              + " -- differences among these are within noise.")
    else:
        print("  Clearly ahead of every other combination on the score.")
    # Edge = extreme of the values tested ALONG the winner's own row/column (grids may be irregular
    # when several searches are combined).
    same_tv = sorted(d for d, tv in combos if tv == winner["TV_Weight"])
    same_delta = sorted(tv for d, tv in combos if d == winner["Huber_Delta"])
    edges = []
    if winner["Huber_Delta"] in (same_tv[0], same_tv[-1]):
        edges.append("huber_delta")
    if winner["TV_Weight"] in (same_delta[0], same_delta[-1]):
        edges.append("tv_weight")
    if edges:
        print(f"  NOTE: the winner is on the edge of the grid for {' and '.join(edges)} -- the best "
              f"value may lie outside the range searched; consider extending it that way.")

    summary_path = os.path.join(out_dir, f"summary{suffix}.csv")
    ranked.to_csv(summary_path, index=False)
    best_path = os.path.join(out_dir, f"best{suffix}.json")
    with open(best_path, "w") as f:
        json.dump({"huber_delta": float(winner["Huber_Delta"]), "tv_weight": float(winner["TV_Weight"]),
                   "weights": weights, "n_images": len(complete_images),
                   "searches": [os.path.basename(d) for d in grid_dirs], "scale": scale,
                   "tied_with": [list(combos[c]) for c in tied],
                   "metrics": {m: float(winner[m]) for m in METRICS}}, f, indent=1)
    print(f"\nSaved {summary_path} and {best_path}.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, m in zip(axes, ["Score", "PSNR", "SSIM"]):
            grid = summary.pivot(index="Huber_Delta", columns="TV_Weight", values=m).reindex(
                index=deltas, columns=tvs)  # untested cells stay blank
            im = ax.imshow(grid.values, cmap="viridis", origin="lower")
            ax.set_xticks(range(len(tvs)), [f"{v:g}" for v in tvs], rotation=45)
            ax.set_yticks(range(len(deltas)), [f"{v:g}" for v in deltas])
            ax.set_xlabel("tv_weight")
            ax.set_ylabel("huber_delta")
            ax.set_title(m)
            for i in range(len(deltas)):
                for j in range(len(tvs)):
                    if not np.isnan(grid.values[i, j]):
                        ax.text(j, i, f"{grid.values[i, j]:.3g}", ha="center", va="center",
                                color="white", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        plot_path = os.path.join(out_dir, f"grid_heatmaps{suffix}.png")
        fig.savefig(plot_path, dpi=130)
        print(f"Saved {plot_path}.")


if __name__ == "__main__":
    main()
