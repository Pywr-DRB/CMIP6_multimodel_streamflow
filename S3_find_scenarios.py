"""
Select two representative climate scenarios from the CMIP6 ensemble.

Method (settings in config.py):
1. Take the monthly % changes in mean NYC inflow from S2 for the PRMS projections under
   SSP2-4.5 and SSP3-7.0 over 2020-2059, relative to the PRMS Daymet historic run.
2. Drop any GCM with an IQR outlier in any month, and any GCM with a negative annual change.
3. Rank the remaining GCMs by the mean of their 12 monthly % changes.
4. 'low' is the lowest-ranked GCM and 'high' the highest.

Writes the selected monthly profiles, a selection summary and all ranked averages to
stats/<baseline>/selected_scenarios/, and two diagnostic figures to
figures/<baseline>/selected_scenarios/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (NODE, HYDRO_MODEL, SSP_PERIOD, BASELINE_SUBDIR,
                    STATS_DIR, FIGURES_DIR, SCENARIO_COLORS)
from scenario_utils import (select_ensemble, filter_scenarios_by_iqr,
                            filter_scenarios_by_annual_change,
                            calculate_weighted_average_changes)

IQR_THRESHOLD = 1.5
REQUIRE_POSITIVE_ANNUAL_CHANGE = True
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def select_low_high(weighted_averages):
    """Lowest- and highest-ranked scenarios from a Series of weighted averages sorted ascending."""
    n_total = len(weighted_averages)
    return {
        'low': {'name': weighted_averages.index[0],
                'weighted_avg': weighted_averages.iloc[0],
                'rank': 1, 'percentile': 0},
        'high': {'name': weighted_averages.index[-1],
                 'weighted_avg': weighted_averages.iloc[-1],
                 'rank': n_total, 'percentile': 100},
    }


def plot_scenario_selection(df_filtered, weighted_averages, selected, output_dir):
    """Histogram of weighted averages, and selected profiles against the filtered ensemble range."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'wspace': 0.3})

    ax1.hist(weighted_averages.values, bins=30, alpha=0.7, color='gray', edgecolor='black')
    for scenario_type, info in selected.items():
        ax1.axvline(info['weighted_avg'], color=SCENARIO_COLORS[scenario_type],
                    linewidth=2.5, linestyle='--',
                    label=f"{scenario_type.capitalize()}: {info['weighted_avg']:.1f}%")
    ax1.set_xlabel('Weighted average flow change (%)', fontsize=12)
    ax1.set_ylabel('Number of scenarios', fontsize=12)
    ax1.set_title(f'(a) Weighted average changes, {len(weighted_averages)} scenarios',
                  fontsize=13, loc='left')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)

    months = range(1, 13)
    ax2.fill_between(months, df_filtered.min(axis=1).values, df_filtered.max(axis=1).values,
                     alpha=0.3, color='gray', label='Range of filtered scenarios')
    ax2.plot(months, df_filtered.mean(axis=1).values, 'k--', linewidth=2,
             label='Mean of filtered scenarios', alpha=0.7)
    for scenario_type, info in selected.items():
        ax2.plot(months, df_filtered[info['name']].values, marker='o', linewidth=3,
                 markersize=9, label=f"{scenario_type.capitalize()} (selected)",
                 color=SCENARIO_COLORS[scenario_type])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Flow change (%)', fontsize=12)
    ax2.set_title('(b) Selected scenarios within the filtered ensemble', fontsize=13, loc='left')
    ax2.set_xticks(list(months))
    ax2.set_xticklabels(MONTH_NAMES)
    ax2.legend(fontsize=9, loc='best', ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Scenario selection: {NODE} | {HYDRO_MODEL} | {SSP_PERIOD}', fontsize=15, y=1.02)
    fname = f'{output_dir}/{NODE}_scenario_selection_{HYDRO_MODEL}_{SSP_PERIOD}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fname}")


def plot_rank_distribution(weighted_averages, selected, output_dir):
    """Bar chart of ranked weighted averages with the selected scenarios annotated."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ranks = np.arange(1, len(weighted_averages) + 1)
    colors_bar = ['lightgray'] * len(weighted_averages)
    for scenario_type, info in selected.items():
        colors_bar[info['rank'] - 1] = SCENARIO_COLORS[scenario_type]

    ax.bar(ranks, weighted_averages.values, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    for scenario_type, info in selected.items():
        value = info['weighted_avg']
        ax.annotate(f"{scenario_type.capitalize()}\n{info['name']}\n{value:.1f}%",
                    xy=(info['rank'], value), xytext=(0, 20 if value > 0 else -20),
                    textcoords='offset points', ha='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=SCENARIO_COLORS[scenario_type], alpha=0.7),
                    arrowprops=dict(arrowstyle='->', lw=1.5))

    ax.set_xlabel('Scenario rank (by weighted average)', fontsize=12)
    ax.set_ylabel('Weighted average flow change (%)', fontsize=12)
    ax.set_title(f'Ranked scenarios: {NODE} | {HYDRO_MODEL} | {SSP_PERIOD}', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    fname = f'{output_dir}/{NODE}_rank_distribution_{HYDRO_MODEL}_{SSP_PERIOD}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fname}")


def main():
    stats_dir = os.path.join(STATS_DIR, BASELINE_SUBDIR)
    selected_stats_dir = os.path.join(stats_dir, 'selected_scenarios')
    selected_figures_dir = os.path.join(FIGURES_DIR, BASELINE_SUBDIR, 'selected_scenarios')
    os.makedirs(selected_stats_dir, exist_ok=True)
    os.makedirs(selected_figures_dir, exist_ok=True)

    monthly_prc_change = pd.read_csv(
        f'{stats_dir}/{NODE}_monthly_mean_prc_change_by_dataset_ssp_and_period.csv', index_col=0)
    monthly_means = pd.read_csv(f'{STATS_DIR}/datasets_{NODE}_monthly_means.csv', index_col=0)

    ensemble = select_ensemble(monthly_prc_change.columns)
    df = monthly_prc_change[ensemble]
    print(f"Scenario selection for {NODE}: {HYDRO_MODEL}, SSP2-4.5 and SSP3-7.0, {SSP_PERIOD}")
    print(f"Ensemble size: {len(ensemble)}")

    # Filter, then rank by the (equal-weight) mean of the monthly % changes
    df_filtered, _ = filter_scenarios_by_iqr(df, outlier_threshold=IQR_THRESHOLD)
    if REQUIRE_POSITIVE_ANNUAL_CHANGE:
        df_filtered, _ = filter_scenarios_by_annual_change(df_filtered, monthly_means, require_positive=True)

    weighted_averages = calculate_weighted_average_changes(df_filtered).sort_values()
    print(f"\nWeighted averages: min {weighted_averages.min():.2f}%, "
          f"median {weighted_averages.median():.2f}%, max {weighted_averages.max():.2f}%")

    selected = select_low_high(weighted_averages)
    print("\nSelected scenarios:")
    for scenario_type, info in selected.items():
        print(f"  {scenario_type:5s} {info['name']}  ({info['weighted_avg']:.2f}%, "
              f"rank {info['rank']} of {len(weighted_averages)})")

    # Monthly profiles of the selected scenarios (the file used by the downstream study)
    selected_traces = pd.DataFrame({k: df_filtered[v['name']] for k, v in selected.items()})
    selected_traces.index.name = 'month'
    fname = f'{selected_stats_dir}/{NODE}_selected_scenarios_{HYDRO_MODEL}_{SSP_PERIOD}.csv'
    selected_traces.to_csv(fname)
    print(f"\nSelected scenario profiles saved: {fname}")

    summary = pd.DataFrame([{
        'scenario_type': scenario_type,
        'gcm_name': info['name'],
        'weighted_avg_change': info['weighted_avg'],
        'rank': info['rank'],
        'total_scenarios': len(weighted_averages),
        'percentile': info['percentile'],
    } for scenario_type, info in selected.items()])
    fname = f'{selected_stats_dir}/{NODE}_selection_summary_{HYDRO_MODEL}_{SSP_PERIOD}.csv'
    summary.to_csv(fname, index=False)
    print(f"Selection summary saved: {fname}")

    fname = f'{selected_stats_dir}/{NODE}_all_weighted_averages_{HYDRO_MODEL}_{SSP_PERIOD}.csv'
    weighted_averages.to_csv(fname, header=['weighted_avg_change'])
    print(f"All weighted averages saved: {fname}")

    plot_scenario_selection(df_filtered, weighted_averages, selected, selected_figures_dir)
    plot_rank_distribution(weighted_averages, selected, selected_figures_dir)


if __name__ == "__main__":
    main()
