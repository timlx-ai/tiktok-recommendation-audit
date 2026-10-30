"""Descriptive analysis of the archived TikTok political recommendation audit.

Data: Ibrahim et al., Zenodo release 10.5281/zenodo.18907782.
Rates describe retained records, including Neutral stance labels in denominators.
Run from this folder with Python 3.12, NumPy and pandas installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ["GEORGIA", "NY", "TEXAS"]
NAMES = ["Georgia", "New York", "Texas", "Overall"]
CONDITIONS = ["left", "right"]
METRICS = ["aligned", "opposing", "support_aligned", "support_opposing"]
REPUBLICAN = {"Pro Republican", "Anti Democrat"}
DEMOCRATIC = {"Pro Democrat", "Anti Republican"}
LABELS = REPUBLICAN | DEMOCRATIC | {"Neutral"}
KEYS = ["state", "week", "leaning", "device"]


def bootstrap_weights(rng, count, n_weeks, block_length):
    """Joint calendar-week sampling; no independent state/party sampling."""
    if block_length == 1:
        return rng.multinomial(n_weeks, np.full(n_weeks, 1 / n_weeks), count)
    n_blocks = (n_weeks + block_length - 1) // block_length
    starts = rng.integers(0, n_weeks - block_length + 1, (count, n_blocks))
    sampled = (starts[..., None] + np.arange(block_length)).reshape(count, -1)
    sampled = sampled[:, :n_weeks]
    weights = np.zeros((count, n_weeks), dtype=int)
    for i in range(n_weeks):
        weights[:, i] = (sampled == i).sum(axis=1)
    return weights


def weighted_estimates(cells, weights):
    """Weeks x states x conditions x metrics; missing state-weeks stay absent."""
    present = np.isfinite(cells[..., 0])
    denominators = np.einsum("bw,wsg->bsg", weights, present)
    valid = (denominators > 0).all(axis=(1, 2))
    numerators = np.einsum("bw,wsgk->bsgk", weights, np.nan_to_num(cells))
    result = numerators[valid] / denominators[valid, ..., None]
    # Equal-state average follows the within-state equal-observed-week mean.
    return np.concatenate([result, result.mean(axis=1, keepdims=True)], axis=1), valid


def bootstrap(cells, reps, seed, block_length):
    rng = np.random.default_rng(seed)
    kept = []
    discarded = 0
    remaining = reps
    first_valid_weights = None
    while remaining:
        weights = bootstrap_weights(rng, remaining, len(cells), block_length)
        estimates, valid = weighted_estimates(cells, weights)
        discarded += int((~valid).sum())
        if first_valid_weights is None and valid.any():
            first_valid_weights = weights[np.flatnonzero(valid)[0]].tolist()
        kept.append(estimates)
        remaining -= len(estimates)
    return np.concatenate(kept), discarded, first_valid_weights


def independent_check(path, estimated):
    """Independent standard-library aggregation, without pandas grouping."""
    groups = defaultdict(lambda: [0, 0, 0])
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            if row["leaning"] not in CONDITIONS:
                continue
            key = (row["state"], int(row["week"]), row["leaning"], row["device"])
            values = groups[key]
            values[0] += 1
            same = DEMOCRATIC if row["leaning"] == "left" else REPUBLICAN
            other = REPUBLICAN if row["leaning"] == "left" else DEMOCRATIC
            values[1] += row["q3_majority"] in same
            values[2] += row["q3_majority"] in other
    by_cell = defaultdict(list)
    for key, values in groups.items():
        by_cell[key[:3]].append(np.array(values[1:]) / values[0])
    result = np.empty((3, 2, 2))
    for s, state in enumerate(STATES):
        for g, condition in enumerate(CONDITIONS):
            weekly = [np.mean(v, axis=0) for k, v in by_cell.items()
                      if k[0] == state and k[2] == condition]
            result[s, g] = np.mean(weekly, axis=0)
    result = np.concatenate([result, result.mean(axis=0, keepdims=True)])
    np.testing.assert_allclose(result, estimated[:, :, :2], atol=1e-12, rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "data/recommendations.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "analysis_results")
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=4802026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.input.read_bytes()
    # Video/device identifiers remain strings to prevent floating-point ID loss.
    frame = pd.read_csv(args.input, dtype={"video_id": str, "device": str})
    assert len(frame) == 24547
    assert frame.isna().sum().sum() == 0
    assert set(frame.q3_majority) == LABELS
    assert set(frame.collection) == {"forYou"}
    assert frame.duplicated().sum() == 0
    assert frame.video_id.nunique() == 12664
    original_counts = frame.leaning.value_counts().to_dict()
    assert original_counts == {"left": 11736, "right": 10661, "neutral": 2150}
    data = frame.loc[frame.leaning.isin(CONDITIONS)].copy()
    is_left = data.leaning.eq("left")
    data["aligned"] = np.where(is_left, data.q3_majority.isin(DEMOCRATIC), data.q3_majority.isin(REPUBLICAN))
    data["opposing"] = np.where(is_left, data.q3_majority.isin(REPUBLICAN), data.q3_majority.isin(DEMOCRATIC))
    data["support_aligned"] = np.where(is_left, data.q3_majority.eq("Pro Democrat"), data.q3_majority.eq("Pro Republican"))
    data["support_opposing"] = np.where(is_left, data.q3_majority.eq("Pro Republican"), data.q3_majority.eq("Pro Democrat"))
    assert ((data.aligned.astype(int) + data.opposing.astype(int)) <= 1).all()
    run_means = data.groupby(KEYS, sort=True)[METRICS].mean()
    run_sizes = data.groupby(KEYS, sort=True).size().rename("n_records")
    runs = run_means.join(run_sizes).reset_index()
    runs.to_csv(args.output / "run_level_shares.csv", index=False)
    weekly = runs.groupby(["state", "week", "leaning"], sort=True)[METRICS].mean()
    weekly.to_csv(args.output / "state_week_shares.csv")
    weeks = list(range(1, 28))
    cells = np.full((27, 3, 2, 4), np.nan)
    for s, state in enumerate(STATES):
        for g, condition in enumerate(CONDITIONS):
            for w, week in enumerate(weeks):
                key = (state, week, condition)
                if key in weekly.index:
                    cells[w, s, g] = weekly.loc[key].to_numpy()
    np.testing.assert_array_equal(np.isfinite(cells[:, :, 0]), np.isfinite(cells[:, :, 1]))
    observed_weeks = np.isfinite(cells[:, :, 0, 0]).sum(axis=0)
    assert observed_weeks.tolist() == [21, 26, 18]
    point = weighted_estimates(cells, np.ones((1, 27), dtype=int))[0][0]
    independent_check(args.input, point)
    boot, rejected, first_weights = bootstrap(cells, args.replicates, args.seed, 1)
    block_boot, block_rejected, _ = bootstrap(cells, args.replicates, args.seed + 1, 2)
    # Verify the first joint-week replicate with explicit loops.
    loop = np.empty((3, 2, 4))
    for s in range(3):
        for g in range(2):
            chosen = [cells[w, s, g] for w, n in enumerate(first_weights)
                      if np.isfinite(cells[w, s, g, 0]) for _ in range(n)]
            loop[s, g] = np.mean(chosen, axis=0)
    loop = np.concatenate([loop, loop.mean(axis=0, keepdims=True)])
    np.testing.assert_allclose(loop, boot[0], atol=1e-12, rtol=0)
    rows = []
    for s, name in enumerate(NAMES):
        sub = data if s == 3 else data.loc[data.state.eq(STATES[s])]
        run_sub = runs if s == 3 else runs.loc[runs.state.eq(STATES[s])]
        for k, metric in enumerate(METRICS):
            gap_boot = 100 * (boot[:, s, 1, k] - boot[:, s, 0, k])
            gap_block = 100 * (block_boot[:, s, 1, k] - block_boot[:, s, 0, k])
            lower, upper = np.quantile(gap_boot, [0.025, 0.975])
            block_lower, block_upper = np.quantile(gap_block, [0.025, 0.975])
            pooled = sub.groupby("leaning")[metric].mean()
            rows.append({
                "location": name, "metric": metric,
                "n_D": int(sub.leaning.eq("left").sum()), "n_R": int(sub.leaning.eq("right").sum()),
                "runs_D": int(run_sub.leaning.eq("left").sum()), "runs_R": int(run_sub.leaning.eq("right").sum()),
                "weeks": None if s == 3 else int(observed_weeks[s]),
                "D_percent": float(100 * point[s, 0, k]), "R_percent": float(100 * point[s, 1, k]),
                "gap_R_minus_D_pp": float(100 * (point[s, 1, k] - point[s, 0, k])),
                "ci95_lower_pp": float(lower), "ci95_upper_pp": float(upper),
                "block2_ci95_lower_pp": float(block_lower), "block2_ci95_upper_pp": float(block_upper),
                "pooled_D_percent": float(100 * pooled["left"]),
                "pooled_R_percent": float(100 * pooled["right"]),
                "pooled_gap_R_minus_D_pp": float(100 * (pooled["right"] - pooled["left"])),
            })
    results = pd.DataFrame(rows)
    results.to_csv(args.output / "results_full_precision.csv", index=False)
    provenance = {
        "source_doi": "10.5281/zenodo.18907782", "archive_date": "2026-03-08",
        "archive_commit": "dd005d3", "input_sha256": hashlib.sha256(raw).hexdigest(),
        "source_records": len(frame), "source_unique_video_ids": frame.video_id.nunique(),
        "record_counts": original_counts, "primary_records": len(data), "primary_runs": len(runs),
        "observed_weeks": dict(zip(STATES, map(int, observed_weeks))),
        "n_state_week_pairs": int(np.isfinite(cells[:, :, 0, 0]).sum()),
        "estimand": "equal-state, equal-observed-week mean retained-record exposure contrast",
        "unit": "percentage points, Republican-conditioned minus Democratic-conditioned",
        "bootstrap": {"replicates": args.replicates, "seed": args.seed, "rejected_no_state": rejected,
                      "unit": "calendar week, joint over all available states and conditions",
                      "interval": "2.5th and 97.5th percentiles of bootstrap contrasts"},
        "adjacent_week_sensitivity": {"block_length": 2, "type": "non-circular moving blocks of consecutive calendar weeks",
                                      "replicates": args.replicates, "seed": args.seed + 1,
                                      "rejected_no_state": block_rejected},
        "unresolved": ["Q1/Q2 filtering", "per-model votes and unresolved-consensus handling",
                       "transcript and advertisement exclusion lineage", "raw run pairing and truncation",
                       "linkage between released records and final published analytical populations"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "checks": ["Counts/schema/labels/missingness/duplicate-row validation",
                   "No video-ID deduplication", "All observed state-weeks contain both partisan conditions",
                   "Independent csv-based aggregation agrees with pandas/numpy point estimates",
                   "First joint-week replicate agrees with explicit repeated-week calculation"],
    }
    (args.output / "provenance_and_methods.json").write_text(json.dumps(provenance, indent=2))
    print(results.loc[results.metric.isin(["aligned", "opposing"]),
                      ["location", "metric", "D_percent", "R_percent", "gap_R_minus_D_pp",
                       "ci95_lower_pp", "ci95_upper_pp"]].round(4).to_string(index=False))
    print("\nRobustness, overall:")
    print(results.loc[results.location.eq("Overall"), ["metric", "gap_R_minus_D_pp",
                     "block2_ci95_lower_pp", "block2_ci95_upper_pp", "pooled_gap_R_minus_D_pp"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
