"""
Two-panel figure of the scenarios selected by S3:
(a) ranked mean annual flow change of every projection in the ensemble, with the selected
    scenarios highlighted, and
(b) monthly % changes of the selected scenarios against the full ensemble range.

Reads S2 and S3 outputs from stats/ and writes
figures/<baseline>/selected_scenarios/<node>_selected_scenarios_<model>_<period>.png.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (NODE, HYDRO_MODEL, SSP_PERIOD, BASELINE_SUBDIR, STATS_DIR, FIGURES_DIR,
                    SCENARIO_COLORS, SCENARIO_LABELS)
from scenario_utils import select_ensemble, calculate_weighted_average_changes

ENSEMBLE_LABEL = 'Range of PRMS CMIP6 projections (SSP2-4.5 and SSP3-7.0)'
ENSEMBLE_COLOR = '#808080'
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def plot_selected_scenarios(selected_df, selected_names, ensemble_df, output_dir):
    """
    selected_df:    monthly % change of each selected scenario (columns 'low', 'high')
    selected_names: {scenario_type: dataset name}
    ensemble_df:    monthly % change of every projection in the ensemble (before filtering)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={'width_ratios': [1, 1.67], 'wspace': 0.3})

    # (a) ranked annual changes
    sorted_averages = calculate_weighted_average_changes(ensemble_df).sort_values()
    ranks = np.arange(1, len(sorted_averages) + 1)
    ax1.plot(ranks, sorted_averages.values, color=ENSEMBLE_COLOR, linewidth=2,
             marker='o', markersize=6, alpha=0.7, label=ENSEMBLE_LABEL, zorder=1)
    for scenario_type, name in selected_names.items():
        rank = sorted_averages.index.get_loc(name) + 1
        ax1.plot(rank, sorted_averages[name], marker='o', markersize=10,
                 color=SCENARIO_COLORS[scenario_type], zorder=10,
                 label=SCENARIO_LABELS[scenario_type])
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Scenario rank', fontsize=11)
    ax1.set_ylabel('Change in mean annual flow (%)', fontsize=11)
    ax1.set_title('(a) Ranked annual flow changes', fontsize=12, loc='left')
    ax1.grid(True, alpha=0.3, axis='y')

    # (b) monthly changes
    months = range(1, 13)
    ax2.fill_between(months, ensemble_df.min(axis=1).values, ensemble_df.max(axis=1).values,
                     alpha=0.4, color=ENSEMBLE_COLOR, label=ENSEMBLE_LABEL)
    for scenario_type in selected_names:
        ax2.plot(months, selected_df[scenario_type].values, marker='o', linewidth=2.5,
                 markersize=6, color=SCENARIO_COLORS[scenario_type],
                 label=SCENARIO_LABELS[scenario_type])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Month', fontsize=11)
    ax2.set_ylabel('Change in mean monthly flow (%)', fontsize=11)
    ax2.set_title('(b) Monthly flow changes', fontsize=12, loc='left')
    ax2.set_xticks(list(months))
    ax2.set_xticklabels(MONTH_LABELS, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # One legend below both panels: ensemble first, then scenarios, no duplicates
    handles, labels = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    legend = {}
    for h, l in zip(handles + h2, labels + l2):
        legend.setdefault(l, h)
    order = [ENSEMBLE_LABEL] + [SCENARIO_LABELS[s] for s in selected_names]
    fig.legend([legend[l] for l in order], order, loc='upper center',
               bbox_to_anchor=(0.5, -0.02), ncol=len(order), fontsize=10, frameon=True)

    fname = f'{output_dir}/{NODE}_selected_scenarios_{HYDRO_MODEL}_{SSP_PERIOD}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def main():
    stats_dir = os.path.join(STATS_DIR, BASELINE_SUBDIR)
    selected_stats_dir = os.path.join(stats_dir, 'selected_scenarios')
    figures_dir = os.path.join(FIGURES_DIR, BASELINE_SUBDIR, 'selected_scenarios')
    os.makedirs(figures_dir, exist_ok=True)

    selected_file = f'{selected_stats_dir}/{NODE}_selected_scenarios_{HYDRO_MODEL}_{SSP_PERIOD}.csv'
    summary_file = f'{selected_stats_dir}/{NODE}_selection_summary_{HYDRO_MODEL}_{SSP_PERIOD}.csv'
    if not (os.path.exists(selected_file) and os.path.exists(summary_file)):
        print(f"S3 outputs not found in {selected_stats_dir}. Run S3_find_scenarios.py first.")
        return

    selected_df = pd.read_csv(selected_file, index_col=0)
    summary = pd.read_csv(summary_file)
    selected_names = dict(zip(summary['scenario_type'], summary['gcm_name']))

    monthly_prc_change = pd.read_csv(
        f'{stats_dir}/{NODE}_monthly_mean_prc_change_by_dataset_ssp_and_period.csv', index_col=0)
    ensemble_df = monthly_prc_change[select_ensemble(monthly_prc_change.columns)]
    print(f"Ensemble of {ensemble_df.shape[1]} projections; selected: {selected_names}")

    plot_selected_scenarios(selected_df, selected_names, ensemble_df, figures_dir)


if __name__ == "__main__":
    main()
