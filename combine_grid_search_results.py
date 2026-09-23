#!/usr/bin/env python3
"""Merge per-slice grid-search outputs (run_knee_grid_search_g1..g5.sbatch, and the edge-extended
run_knee_grid_search_e1..e5.sbatch re-run) into ONE combined trial-history CSV and ONE combined
best-params JSON.

Run this AFTER the jobs you want merged have finished. Each job wrote its own tagged files
(optuna_best_huber_tv_accel_*_g1.json / _e1.json / etc, and matching trial-history CSVs) rather
than sharing one file live -- concurrent processes writing the same path would race and silently
lose whichever finished last. This script is the safe way to get back down to one file per output
once there's no more concurrent writing happening. Default --tags covers g1,g2,g3,g5 (g4 never
completed in the first pass) plus e1-e5 (the edge-extended re-run) -- pass --tags explicitly to
merge a different subset.

Usage:
    python3 combine_grid_search_results.py [--ckpt-root ~/fastmri_results/knee] [--tags g1,g2,...]
"""
import argparse
import glob
import json
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-root", default=os.environ.get("CONVDECODER_CKPT_ROOT", "~/fastmri_results/knee"),
        help="Same CKPT_ROOT the sbatch scripts used (default: $CONVDECODER_CKPT_ROOT or "
             "~/fastmri_results/knee).",
    )
    parser.add_argument(
        "--tags", default="g1,g2,g3,g5,e1,e2,e3,e4,e5",
        help="Comma-separated GRID_SEARCH_JOB_TAG values to merge (default covers the first pass's "
             "completed slices -- g4 never finished -- plus the edge-extended e1-e5 re-run).",
    )
    args = parser.parse_args()

    ckpt_root = os.path.expanduser(args.ckpt_root)
    tuning_dir = os.path.join(ckpt_root, "tuning")
    tags = args.tags.split(",")

    # --- 1. Merge trial-history CSVs ---
    # Named "..._full_..." not "..._accel_..." -- run_hyperparameter_search's trials_csv_path uses
    # _regime_suffix = "accel" if guided_init else "full", and this grid search deliberately runs
    # with guided_init=False (Run A's un-guided regime), so the CSVs land under the "full" tag
    # despite still searching ACCEL_HUBER_DELTA/ACCEL_TV_WEIGHT. The best-params JSON is unaffected
    # by this quirk -- ACCEL_OPTUNA_BEST_PATH's own filename is always "accel"-tagged regardless of
    # guided_init, so that glob below stays as "accel".
    all_trials = []
    for tag in tags:
        pattern = os.path.join(tuning_dir, f"optuna_trials_huber_tv_full_*_{tag}.csv")
        matches = glob.glob(pattern)
        if not matches:
            print(f"WARNING: no trial-history CSV found for tag {tag!r} (pattern: {pattern})")
            continue
        if len(matches) > 1:
            print(f"WARNING: multiple matches for tag {tag!r}, using the most recently modified: "
                  f"{matches}")
            matches.sort(key=os.path.getmtime, reverse=True)
        df = pd.read_csv(matches[0])
        df["source_tag"] = tag
        all_trials.append(df)
        print(f"  loaded {len(df)} trial(s) from {matches[0]}")

    if not all_trials:
        raise SystemExit("No trial-history CSVs found for any tag -- have the g1-g5 jobs "
                          "finished yet?")

    combined_df = pd.concat(all_trials, ignore_index=True)
    # Trial numbers collide across files (each study started its own numbering from 0) --
    # renumber so the combined file has unique, orderable trial identifiers.
    combined_df["combined_trial_number"] = range(len(combined_df))

    # Infer ARCHITECTURE/Z_SOURCE from one of the matched filenames so the combined file's name
    # follows the same convention as the per-slice files.
    example_name = os.path.basename(matches[0])
    # optuna_trials_huber_tv_full_{ARCHITECTURE}_{Z_SOURCE}_{tag}.csv
    stem = example_name[len("optuna_trials_huber_tv_full_"):-len(f"_{tag}.csv")]
    combined_csv_path = os.path.join(tuning_dir, f"optuna_trials_huber_tv_full_{stem}_combined.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"\nWrote combined trial history ({len(combined_df)} trials across {len(all_trials)} "
          f"slice(s)) to {combined_csv_path}")

    # --- 2. Pick the overall winner and write ONE combined best-params JSON ---
    complete = combined_df[combined_df["state"] == "COMPLETE"]
    if complete.empty:
        print("WARNING: no COMPLETE trials found -- skipping combined best-params JSON.")
    else:
        best_row = complete.loc[complete["value"].idxmax()]
        best_record = {
            "huber_delta": float(best_row["params_huber_delta"]),
            "tv_weight": float(best_row["params_tv_weight"]),
            "value": float(best_row["value"]),
            "source_tag": best_row["source_tag"],
            "combined_from_tags": tags,
            "note": "Combined winner across all grid slices, picked by raw grid VALUE (not a "
                    "re-run confirmation pass) -- see each slice's own JSON for its "
                    "confirm_score if you need the confirmed (longer-fit) ranking instead.",
        }
        combined_json_path = os.path.join(ckpt_root, f"optuna_best_huber_tv_accel_{stem}_combined.json")
        with open(combined_json_path, "w") as f:
            json.dump(best_record, f, indent=2)
        print(f"Wrote combined best-params JSON to {combined_json_path}")
        print(f"\nOverall winner: huber_delta={best_record['huber_delta']:.4g}  "
              f"tv_weight={best_record['tv_weight']:.4g}  value={best_record['value']:.4f}  "
              f"(from slice {best_record['source_tag']})")

    # --- 3. Top 5 for a quick sanity look ---
    print("\nTop 5 combinations overall:")
    top5 = complete.sort_values("value", ascending=False).head(5)
    for _, r in top5.iterrows():
        print(f"  value={r['value']:.4f}  delta={r['params_huber_delta']:.4g}  "
              f"tv={r['params_tv_weight']:.4g}  (slice {r['source_tag']})")


if __name__ == "__main__":
    main()
