"""
Functions shared by the scenario-selection scripts (S3, S4, S5).
"""
import numpy as np
import pandas as pd
from config import HYDRO_MODEL, SSP_PERIOD, SSPS
from utils import dataset_baselines


def select_ensemble(columns, hydro_model=HYDRO_MODEL, ssp_period=SSP_PERIOD, ssps=SSPS):
    """Dataset names in `columns` matching the chosen hydrologic model, SSPs and period."""
    return [c for c in columns
            if hydro_model in c and any(ssp in c for ssp in ssps) and ssp_period in c]


def calculate_annual_pct_changes(monthly_means_df, scenarios):
    """
    Percent change in annual flow (sum of the 12 monthly means) for each scenario,
    relative to that scenario's own historic baseline.
    """
    annual_pct_changes = {}
    for scenario in scenarios:
        baseline_name = dataset_baselines.get(scenario, 'pub_nhmv10_BC_withObsScaled')
        if baseline_name not in monthly_means_df.columns:
            print(f"Warning: Baseline '{baseline_name}' not found for '{scenario}'")
            continue
        scenario_annual = monthly_means_df[scenario].sum()
        baseline_annual = monthly_means_df[baseline_name].sum()
        annual_pct_changes[scenario] = ((scenario_annual - baseline_annual) / baseline_annual) * 100
    return pd.Series(annual_pct_changes)


def filter_scenarios_by_iqr(df, outlier_threshold=1.5, verbose=True):
    """
    Drop any scenario (column) that is an IQR outlier in at least one month (row).
    Returns the filtered DataFrame and the list of rejected scenario names.
    """
    outlier_scenarios = set()
    outlier_details = {}

    for month in df.index:
        month_data = df.loc[month].values
        q1 = np.percentile(month_data, 25)
        q3 = np.percentile(month_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - outlier_threshold * iqr
        upper_bound = q3 + outlier_threshold * iqr

        for scenario_name in df.columns:
            value = df.loc[month, scenario_name]
            if value < lower_bound or value > upper_bound:
                outlier_scenarios.add(scenario_name)
                outlier_details.setdefault(scenario_name, []).append(
                    f"month {month}: {value:.1f}% (bounds {lower_bound:.1f} to {upper_bound:.1f})")

    valid_scenarios = [col for col in df.columns if col not in outlier_scenarios]
    df_filtered = df[valid_scenarios].copy()

    if verbose:
        print(f"\nIQR filter (threshold = {outlier_threshold}): "
              f"{len(outlier_scenarios)} of {len(df.columns)} scenarios rejected")
        for scenario in sorted(outlier_scenarios):
            print(f"  {scenario}")
            for detail in outlier_details[scenario]:
                print(f"    {detail}")

    return df_filtered, list(outlier_scenarios)


def filter_scenarios_by_annual_change(df, monthly_means_df, require_positive=True, verbose=True):
    """
    Drop scenarios whose annual flow change is negative (when require_positive is True).
    Returns the filtered DataFrame and a list of (name, annual % change) for rejected scenarios.
    """
    scenarios_to_keep = []
    scenarios_rejected = []
    annual_changes = calculate_annual_pct_changes(monthly_means_df, df.columns.tolist())

    for scenario in df.columns:
        if scenario not in annual_changes.index:
            scenarios_to_keep.append(scenario)
            continue
        annual_pct_change = annual_changes[scenario]
        if require_positive and annual_pct_change < 0:
            scenarios_rejected.append((scenario, annual_pct_change))
        else:
            scenarios_to_keep.append(scenario)

    df_filtered = df[scenarios_to_keep]

    if verbose:
        print(f"\nAnnual change filter (require_positive={require_positive}): "
              f"{len(scenarios_rejected)} rejected, {len(scenarios_to_keep)} remaining")
        for scenario, pct_change in scenarios_rejected:
            print(f"  {scenario}: {pct_change:.2f}%")

    return df_filtered, scenarios_rejected


def calculate_weighted_average_changes(monthly_pct_change_df, weights=None):
    """
    Weighted average of the monthly % changes for each scenario (column).
    Equal weights when `weights` is None.
    """
    if weights is None:
        weights = np.ones(len(monthly_pct_change_df))
    else:
        weights = np.array(weights)
    weights = weights / weights.sum()

    weighted_avg = {}
    for scenario in monthly_pct_change_df.columns:
        values = monthly_pct_change_df[scenario].values
        weighted_avg[scenario] = np.sum(values * weights)

    return pd.Series(weighted_avg)
