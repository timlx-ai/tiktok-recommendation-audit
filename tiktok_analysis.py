import os
import numpy as np
import pandas as pd

# TikTok Political Recommendation Audit Replication (Ibrahim et al., 2026)
# Author: Tim Laux


def load_data(csv_path):
    """Load recommendations dataset and filter for Democratic/Republican runs."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Missing data file at {csv_path}. Make sure recommendations.csv is in the data/ folder."
        )

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Recode leaning column to standard party labels
    party_map = {"left": "Democratic", "right": "Republican"}
    df["condition"] = df["leaning"].map(party_map)

    # Keep only partisan conditioned accounts (drop neutral/unconditioned)
    df_clean = df[df["condition"].isin(["Democratic", "Republican"])].copy()

    # Create unique run identifier from device and collection session
    if "device" in df_clean.columns and "collection" in df_clean.columns:
        df_clean["run_id"] = (
            df_clean["device"].astype(str)
            + "_"
            + df_clean["collection"].astype(str)
        )
    else:
        df_clean["run_id"] = "run_1"

    return df_clean


def add_stance_indicators(df):
    """Code stance alignment relative to account party condition."""

    def is_aligned(row):
        cond, stance = row["condition"], str(row["q3_majority"])
        if cond == "Democratic" and stance in [
            "Pro Democrat",
            "Anti Republican",
        ]:
            return 1
        if cond == "Republican" and stance in [
            "Pro Republican",
            "Anti Democrat",
        ]:
            return 1
        return 0

    def is_opposing(row):
        cond, stance = row["condition"], str(row["q3_majority"])
        if cond == "Democratic" and stance in [
            "Pro Republican",
            "Anti Democrat",
        ]:
            return 1
        if cond == "Republican" and stance in [
            "Pro Democrat",
            "Anti Republican",
        ]:
            return 1
        return 0

    def is_supportive_only(row):
        cond, stance = row["condition"], str(row["q3_majority"])
        if cond == "Democratic" and stance == "Pro Democrat":
            return 1
        if cond == "Republican" and stance == "Pro Republican":
            return 1
        return 0

    df["aligned"] = df.apply(is_aligned, axis=1)
    df["opposing"] = df.apply(is_opposing, axis=1)
    df["supportive_only"] = df.apply(is_supportive_only, axis=1)
    return df


def compute_weighted_gap(df, col_name):
    """
    4-layer weighting hierarchy:
    1. Run-level average
    2. State-Week cell average
    3. Equal week average per state
    4. Equal 1/3 weight across the 3 states (GA, NY, TX)
    """
    run_avg = (
        df.groupby(["state", "week", "condition", "run_id"])[col_name]
        .mean()
        .reset_index()
    )
    cell_avg = (
        run_avg.groupby(["state", "week", "condition"])[col_name]
        .mean()
        .reset_index()
    )
    state_avg = (
        cell_avg.groupby(["state", "condition"])[col_name].mean().reset_index()
    )

    overall = state_avg.groupby("condition")[col_name].mean()

    dem_pct = overall.get("Democratic", 0) * 100
    rep_pct = overall.get("Republican", 0) * 100
    gap = rep_pct - dem_pct

    return dem_pct, rep_pct, gap


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_dir, "data", "recommendations.csv")

    print(f"Reading dataset from {data_file}...")
    data = load_data(data_file)
    data = add_stance_indicators(data)

    print(f"Loaded {len(data):,} partisan records.")

    # Calculate point estimates
    dem_aligned, rep_aligned, gap_aligned = compute_weighted_gap(
        data, "aligned"
    )
    dem_opp, rep_opp, gap_opp = compute_weighted_gap(data, "opposing")
    dem_supp, rep_supp, gap_supp = compute_weighted_gap(data, "supportive_only")

    # Calendar-week block bootstrap (20,000 iterations)
    print("Running 20,000 week-block bootstrap resamples...")
    np.random.seed(42)
    unique_weeks = data["week"].dropna().unique()
    boot_gaps = []

    for _ in range(20000):
        sampled_weeks = np.random.choice(
            unique_weeks, size=len(unique_weeks), replace=True
        )
        boot_df = pd.concat([data[data["week"] == w] for w in sampled_weeks])
        _, _, b_gap = compute_weighted_gap(boot_df, "aligned")
        boot_gaps.append(b_gap)

    ci_low, ci_high = np.percentile(boot_gaps, [2.5, 97.5])

    # Output summary to console
    print("\n--- Summary Results ---")
    print(
        f"Aligned Exposure Share:  Dem = {dem_aligned:.2f}%, Rep = {rep_aligned:.2f}%"
    )
    print(f"Aligned Gap:             {gap_aligned:+.2f} percentage points")
    print(f"95% Bootstrap CI:        [{ci_low:.2f}, {ci_high:.2f}] pp")
    print(f"Opposing Exposure Gap:   {gap_opp:+.2f} percentage points")
    print(f"Supportive-Only Check:   {gap_supp:+.2f} percentage points")

    # Export full precision values to CSV
    out_dir = os.path.join(base_dir, "analysis_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results_full_precision.csv")

    results_table = pd.DataFrame([
        {
            "Metric": "Aligned Exposure Share",
            "Democratic_Rate": dem_aligned,
            "Republican_Rate": rep_aligned,
            "Gap_pp": gap_aligned,
            "CI_95_Lower": ci_low,
            "CI_95_Upper": ci_high,
        },
        {
            "Metric": "Opposing Exposure Share",
            "Democratic_Rate": dem_opp,
            "Republican_Rate": rep_opp,
            "Gap_pp": gap_opp,
            "CI_95_Lower": np.nan,
            "CI_95_Upper": np.nan,
        },
        {
            "Metric": "Supportive-Only Check",
            "Democratic_Rate": dem_supp,
            "Republican_Rate": rep_supp,
            "Gap_pp": gap_supp,
            "CI_95_Lower": np.nan,
            "CI_95_Upper": np.nan,
        },
    ])
    results_table.to_csv(out_path, index=False)
    print(f"Saved results file to {out_path}")


if __name__ == "__main__":
    main()
