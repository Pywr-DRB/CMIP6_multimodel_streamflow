"""
Annual flow change of every projection in the ensemble relative to its own baseline,
with the scenarios selected by S3 marked:
(a) box plot with individual projections, and
(b) empirical CDF with the percentile of each selected scenario.

Reads S2 and S3 outputs from stats/ and writes
figures/<baseline>/selected_scenarios/<node>_annual_flow_comparison_<model>_<period>.png.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (NODE, HYDRO_MODEL, SSP_PERIOD, BASELINE_SUBDIR, STATS_DIR, FIGURES_DIR,
                    SCENARIO_COLORS)
from scenario_utils import select_ensemble, calculate_annual_pct_changes


def ordinal(n):
    suffix = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


def plot_annual_flow_comparison(ensemble_pct_change, selected_pct_change, output_dir):
    """
    ensemble_pct_change: Series of annual % change for every projection
    selected_pct_change: {scenario_type: annual % change}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'wspace': 0.25})
    values = ensemble_pct_change.values

    # (a) distribution
    ax1.boxplot(values, positions=[1], widths=0.5, patch_artist=True,
                boxprops=dict(facecolor='lightgray', alpha=0.6),
                medianprops=dict(color='black', linewidth=2.5),
                whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))
    rng = np.random.default_rng(0)
    ax1.scatter(rng.normal(1, 0.04, size=len(values)), values, alpha=0.3, s=30, color='gray', zorder=1)
    for scenario_type, pct_change in selected_pct_change.items():
        ax1.scatter([1], [pct_change], s=300, color=SCENARIO_COLORS[scenario_type],
                    edgecolors='black', linewidth=2.5, zorder=10, marker='D')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
    ax1.set_xlim(0.5, 1.5)
    ax1.set_xticks([])
    ax1.set_ylabel('Change in annual mean flow (%)\nrelative to dataset baseline', fontsize=12)
    ax1.set_title('(a) Distribution of annual flow changes', fontsize=13, loc='left')
    ax1.grid(True, alpha=0.3, axis='y')

    q25, q75 = np.percentile(values, [25, 75])
    stats_text = (f"N = {len(values)} projections\n"
                  f"Median: {np.median(values):+.1f}%\n"
                  f"IQR: {q25:+.1f}% to {q75:+.1f}%\n"
                  f"Range: {values.min():+.1f}% to {values.max():+.1f}%")
    ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    # (b) empirical CDF
    sorted_values = np.sort(values)
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values) * 100
    ax2.plot(sorted_values, cdf, linewidth=3, color='gray', label='GCM ensemble', zorder=1)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1.5, label='No change', alpha=0.7, zorder=2)
    for scenario_type, pct_change in selected_pct_change.items():
        percentile = (sorted_values < pct_change).sum() / len(sorted_values) * 100
        ax2.axvline(pct_change, color=SCENARIO_COLORS[scenario_type], linewidth=2.5,
                    alpha=0.8, linestyle=':', zorder=3)
        ax2.scatter([pct_change], [percentile], s=300, color=SCENARIO_COLORS[scenario_type],
                    edgecolors='black', linewidth=2.5, zorder=10, marker='D',
                    label=f'{scenario_type.capitalize()} ({ordinal(round(percentile))} percentile)')
    ax2.set_xlabel('Change in annual mean flow (%)', fontsize=12)
    ax2.set_ylabel('Cumulative probability (%)', fontsize=12)
    ax2.set_title('(b) Cumulative distribution of annual changes', fontsize=13, loc='left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    handles, labels = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    legend = {}
    for h, l in zip(handles + h2, labels + l2):
        legend.setdefault(l, h)
    fig.legend(legend.values(), legend.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=5, fontsize=10, frameon=True, edgecolor='black')
    fig.suptitle(f'{NODE}: annual flow changes relative to dataset baselines | '
                 f'{HYDRO_MODEL} | {SSP_PERIOD.replace("_", "-")}', fontsize=14, y=1.00)

    fname = f'{output_dir}/{NODE}_annual_flow_comparison_{HYDRO_MODEL}_{SSP_PERIOD}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def main():
    selected_stats_dir = os.path.join(STATS_DIR, BASELINE_SUBDIR, 'selected_scenarios')
    figures_dir = os.path.join(FIGURES_DIR, BASELINE_SUBDIR, 'selected_scenarios')
    os.makedirs(figures_dir, exist_ok=True)

    summary_file = f'{selected_stats_dir}/{NODE}_selection_summary_{HYDRO_MODEL}_{SSP_PERIOD}.csv'
    if not os.path.exists(summary_file):
        print(f"{summary_file} not found. Run S3_find_scenarios.py first.")
        return

    monthly_means = pd.read_csv(f'{STATS_DIR}/datasets_{NODE}_monthly_means.csv', index_col=0)
    ensemble = select_ensemble(monthly_means.columns)
    ensemble_pct_change = calculate_annual_pct_changes(monthly_means, ensemble)
    print(f"Annual change across {len(ensemble)} projections: median {ensemble_pct_change.median():+.1f}%, "
          f"range {ensemble_pct_change.min():+.1f}% to {ensemble_pct_change.max():+.1f}%")

    summary = pd.read_csv(summary_file)
    selected_pct_change = {row['scenario_type']: ensemble_pct_change[row['gcm_name']]
                           for _, row in summary.iterrows()
                           if row['gcm_name'] in ensemble_pct_change.index}
    for scenario_type, pct_change in selected_pct_change.items():
        print(f"  {scenario_type:5s} {pct_change:+.1f}%")

    plot_annual_flow_comparison(ensemble_pct_change, selected_pct_change, figures_dir)


if __name__ == "__main__":
    main()
