"""
Script for creating comprehensive visualizations of selected climate scenarios.

This script loads the scenarios selected by S3_find_scenarios.py and creates
multiple visualization types including:
1. Quantile space heatmaps with selected scenario traces
2. Monthly flow change comparisons
3. Scenario envelope plots

The quantile space visualization shows how the three selected scenarios
(low, medium, high) traverse through the distribution of all GCM projections.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quantile_utils import create_quantile_matrix
from plotting_functions import plot_quantile_space_with_selected_scenarios


def plot_scenario_monthly_comparison(selected_scenarios_df,
                                      all_scenarios_df,
                                      node,
                                      hydro_model,
                                      ssp_period,
                                      output_dir):
    """
    Create a plot comparing selected scenarios to the full ensemble.

    Parameters:
    -----------
    selected_scenarios_df : pd.DataFrame
        Selected scenarios (columns: low, medium, high)
    all_scenarios_df : pd.DataFrame
        All filtered scenarios from S3
    node : str
        Node name
    hydro_model : str
        Hydrologic model name (PRMS or VIC)
    ssp_period : str
        SSP period (e.g., '2020_2059')
    output_dir : str
        Output directory for figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot envelope of all scenarios
    all_min = all_scenarios_df.min(axis=1)
    all_max = all_scenarios_df.max(axis=1)
    all_median = all_scenarios_df.median(axis=1)

    ax.fill_between(all_scenarios_df.index, all_min, all_max,
                    alpha=0.2, color='gray', label='Full ensemble range')
    ax.plot(all_scenarios_df.index, all_median, 'k--',
            linewidth=2, label='Ensemble median', alpha=0.7)

    # Plot selected scenarios
    scenario_colors = {
        'low': '#2166ac',
        'medium': '#fee090',
        'high': '#b2182b'
    }

    for scenario_type in selected_scenarios_df.columns:
        color = scenario_colors.get(scenario_type, 'gray')
        ax.plot(selected_scenarios_df.index,
                selected_scenarios_df[scenario_type],
                color=color, linewidth=3, marker='o', markersize=6,
                label=f'{scenario_type.capitalize()} scenario',
                zorder=10)

    # Styling
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Flow Change (%) Relative to Baseline', fontsize=12, fontweight='bold')
    ax.set_title(f'{node.replace("_", " ").title()} - Selected Scenarios vs Full Ensemble\n' +
                 f'{hydro_model} | {ssp_period.replace("_", "-")}',
                 fontsize=14, fontweight='bold')

    # Month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)

    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # Save
    fname = f'{output_dir}/{node}_selected_vs_ensemble_{hydro_model}_{ssp_period}.png'
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {fname}")


def main():
    """
    Main execution function for scenario visualization.
    """
    # Configuration - should match S3_find_scenarios.py settings
    node = 'nyc_inflow'
    use_dataset_baseline = True
    hydro_model_source = 'PRMS'
    ssp_period = '2020_2059'

    print(f"\n{'='*80}")
    print(f"CLIMATE SCENARIO VISUALIZATION")
    print(f"{'='*80}")
    print(f"Node:            {node}")
    print(f"Hydro Model:     {hydro_model_source}")
    print(f"Period:          {ssp_period}")
    print(f"{'='*80}\n")

    # Set up paths
    fdir = './stats/diff_relative_to_dataset_baseline' if use_dataset_baseline else './stats/diff_relative_to_reconstruction'
    stats_dir = f'{fdir}/selected_scenarios'
    figures_dir = './figures/diff_relative_to_dataset_baseline/selected_scenarios' if use_dataset_baseline else './figures/diff_relative_to_reconstruction/selected_scenarios'
    os.makedirs(figures_dir, exist_ok=True)

    # Load selected scenarios from S3 output
    selected_scenarios_file = f'{stats_dir}/{node}_selected_scenarios_{hydro_model_source}_{ssp_period}.csv'

    if not os.path.exists(selected_scenarios_file):
        print(f"ERROR: Selected scenarios file not found: {selected_scenarios_file}")
        print(f"Please run S3_find_scenarios.py first.")
        return

    selected_scenarios_df = pd.read_csv(selected_scenarios_file, index_col=0)
    print(f"Loaded selected scenarios from: {selected_scenarios_file}")
    print(f"  Scenarios: {list(selected_scenarios_df.columns)}")

    # Load full monthly percentage change data
    monthly_prc_change_file = f'{fdir}/{node}_monthly_mean_prc_change_by_dataset_ssp_and_period.csv'
    monthly_prc_change = pd.read_csv(monthly_prc_change_file, index_col=0)

    # Filter for the same datasets used in S3
    filtered_datasets = [d for d in monthly_prc_change.columns
                        if hydro_model_source in d
                        and ('ssp245' in d or 'ssp370' in d)
                        and ssp_period in d]

    all_scenarios_df = monthly_prc_change[filtered_datasets]
    print(f"Loaded {len(filtered_datasets)} GCM scenarios for comparison")

    # Create quantile matrix from all filtered scenarios
    print(f"\nCreating quantile matrix...")
    quantile_matrix = create_quantile_matrix(all_scenarios_df)
    print(f"  Quantile matrix shape: {quantile_matrix.shape}")

    # Create visualizations
    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATIONS")
    print(f"{'='*80}\n")

    # 1. Quantile space visualization with selected scenarios
    print("1. Creating quantile space visualization...")
    quantile_fname = f'{figures_dir}/{node}_quantile_space_selected_{hydro_model_source}_{ssp_period}.png'
    plot_quantile_space_with_selected_scenarios(
        quantile_matrix,
        selected_scenarios_df,
        node,
        quantile_fname
    )
    print(f"   Saved: {quantile_fname}")

    # 2. Monthly comparison plot
    print("2. Creating monthly comparison plot...")
    plot_scenario_monthly_comparison(
        selected_scenarios_df,
        all_scenarios_df,
        node,
        hydro_model_source,
        ssp_period,
        figures_dir
    )

    print(f"\n{'='*80}")
    print(f"VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
