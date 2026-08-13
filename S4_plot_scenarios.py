"""
Script for creating a comprehensive visualization of selected climate scenarios.

This script loads the scenarios selected by S3_find_scenarios.py and creates
a two-panel synthesis figure showing ranked annual flow changes and monthly
flow change patterns.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scenario_utils import calculate_flow_weights, calculate_weighted_average_changes


# =============================================================================
# Module-level style dictionaries for consistent formatting across all figures
# =============================================================================



SCENARIO_LABELS = {
    'low': 'Wetter Winter, Drier Summer',
    'medium': 'Median',
    'high': 'Wetter Winter',
    'historic': 'Historic Baseline'
}

ENSEMBLE_LABEL = 'Range of PRMS SSP2 RCP4.5 CMIP6 Models'

MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
SCENARIO_COLORS = {
    'low': '#ed9f1c',      # Red
    'medium': '#fee090',   # Yellow
    'high': '#009e73',     # Blue
    'historic': '#000000'  # Black
}

def plot_selected_scenarios(selected_scenarios_df,
                            all_scenarios_df,
                            all_scenarios_unfiltered_df,
                            monthly_means_df,
                            node,
                            hydro_model,
                            ssp_period,
                            output_dir,
                            weight_scheme='equal',
                            scenarios_to_show=None):
    """
    Create a two-panel figure showing selected climate scenarios:
    - Panel a: Rank vs magnitude plot of weighted average annual flow changes
    - Panel b: Selected scenarios in context of full ensemble (monthly % changes)

    Parameters:
    -----------
    selected_scenarios_df : pd.DataFrame
        Selected scenarios (columns: low, medium, high) with monthly % changes
    all_scenarios_df : pd.DataFrame
        All filtered scenarios from S3 (post-IQR filtering) with monthly % changes
    all_scenarios_unfiltered_df : pd.DataFrame
        All scenarios before IQR filtering (pre-IQR) for full ensemble shading
    monthly_means_df : pd.DataFrame
        Monthly mean flows (rows=months, columns=datasets)
    node : str
        Node name
    hydro_model : str
        Hydrologic model name (PRMS or VIC)
    ssp_period : str
        SSP period (e.g., '2020_2059')
    output_dir : str
        Output directory for figure
    weight_scheme : str
        Weighting scheme used in S3 ('equal', 'flow_weighted', or 'log_flow_weighted')
    scenarios_to_show : list or None
        List of scenario types to display (e.g., ['low', 'medium', 'high']).
        If None, defaults to all scenarios in selected_scenarios_df.
    """
    # Default to showing all selected scenarios
    if scenarios_to_show is None:
        scenarios_to_show = list(selected_scenarios_df.columns)

    # Calculate weights based on scheme
    if weight_scheme in ['flow_weighted', 'log_flow_weighted']:
        use_log = (weight_scheme == 'log_flow_weighted')
        weights, _ = calculate_flow_weights(monthly_means_df,
                                           baseline_col='pub_nhmv10_BC_withObsScaled',
                                           log_transform=use_log)
    else:
        weights = None

    # Calculate weighted average changes for both filtered and unfiltered scenarios
    weighted_averages_filtered = calculate_weighted_average_changes(all_scenarios_df, weights=weights)
    weighted_averages_unfiltered = calculate_weighted_average_changes(all_scenarios_unfiltered_df, weights=weights)

    # Get weighted averages for selected scenarios
    selected_weighted_avgs = {}
    for scenario_type in selected_scenarios_df.columns:
        scenario_monthly = selected_scenarios_df[scenario_type]
        for gcm_name in all_scenarios_df.columns:
            if np.allclose(all_scenarios_df[gcm_name].values, scenario_monthly.values):
                selected_weighted_avgs[scenario_type] = weighted_averages_filtered[gcm_name]
                break

    # =========================================================================
    # Consistent style settings for both panels
    # =========================================================================
    ENSEMBLE_COLOR = '#808080'
    ENSEMBLE_ALPHA = 0.4

    # Create figure with 2 panels, width ratio 1:1.67
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={'width_ratios': [1, 1.67], 'wspace': 0.3})

    # ===== PANEL a: Rank vs Magnitude plot =====
    # Sort scenarios by weighted average change
    sorted_averages = weighted_averages_unfiltered.sort_values()
    ranks = np.arange(1, len(sorted_averages) + 1)

    # Create line plot for ensemble
    ax1.plot(ranks, sorted_averages.values, color=ENSEMBLE_COLOR, linewidth=2,
             marker='o', markersize=6, alpha=0.7, label=ENSEMBLE_LABEL, zorder=1)

    # Highlight selected scenarios with colored markers
    for scenario_name, rank in zip(sorted_averages.index, ranks):
        for scenario_type in scenarios_to_show:
            if scenario_type in selected_scenarios_df.columns:
                scenario_monthly = selected_scenarios_df[scenario_type]
                if scenario_name in all_scenarios_unfiltered_df.columns:
                    if np.allclose(all_scenarios_unfiltered_df[scenario_name].values, scenario_monthly.values):
                        ax1.plot(rank, sorted_averages[scenario_name],
                                marker='o', markersize=10, color=SCENARIO_COLORS[scenario_type],
                                zorder=10, label=SCENARIO_LABELS[scenario_type])
                        break

    # Add horizontal line at zero
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax1.set_xlabel('Scenario Rank', fontsize=11)
    ax1.set_ylabel('Change in Mean Annual Flow (%)', fontsize=11)
    ax1.set_title('(a) Ranked Annual Flow Changes',
                  fontsize=12, loc='left')
    ax1.grid(True, alpha=0.3, axis='y')

    # ===== PANEL b: Monthly % changes (ensemble context) =====
    # Full ensemble range
    unfiltered_min = all_scenarios_unfiltered_df.min(axis=1)
    unfiltered_max = all_scenarios_unfiltered_df.max(axis=1)
    ax2.fill_between(range(1, 13), unfiltered_min.values, unfiltered_max.values,
                     alpha=ENSEMBLE_ALPHA, color=ENSEMBLE_COLOR,
                     label=ENSEMBLE_LABEL)

    # Overlay selected scenarios (only those in scenarios_to_show)
    for scenario_type in scenarios_to_show:
        if scenario_type in selected_scenarios_df.columns:
            ax2.plot(range(1, 13), selected_scenarios_df[scenario_type].values,
                    marker='o', linewidth=2.5, markersize=6,
                    color=SCENARIO_COLORS[scenario_type],
                    label=SCENARIO_LABELS[scenario_type])

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Month', fontsize=11)
    ax2.set_ylabel('Change in Mean Monthly Flow (%)', fontsize=11)
    ax2.set_title('(b) Monthly Flow Changes',
                  fontsize=12, loc='left')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(MONTH_LABELS, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # ===== Single unified legend below figure =====
    # Collect handles and labels, using consistent naming
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Build legend with consistent labels (no duplicates)
    legend_dict = {}
    for h, l in zip(handles1 + handles2, labels1 + labels2):
        if l not in legend_dict:
            legend_dict[l] = h

    # Order legend items: ensemble first, then scenarios
    ordered_labels = []
    ordered_handles = []
    # Ensemble item first
    if ENSEMBLE_LABEL in legend_dict:
        ordered_labels.append(ENSEMBLE_LABEL)
        ordered_handles.append(legend_dict[ENSEMBLE_LABEL])
    # Then scenario items
    for scenario_type in scenarios_to_show:
        label = SCENARIO_LABELS.get(scenario_type, scenario_type)
        if label in legend_dict:
            ordered_labels.append(label)
            ordered_handles.append(legend_dict[label])

    fig.legend(ordered_handles, ordered_labels,
              loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=len(ordered_labels), fontsize=10, frameon=True)

    # Save
    fname = f'{output_dir}/{node}_selected_scenarios_{hydro_model}_{ssp_period}.png'
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
    weight_scheme = 'equal'  # Should match S3 setting: 'equal', 'flow_weighted', or 'log_flow_weighted'

    print(f"\n{'='*80}")
    print(f"CLIMATE SCENARIO VISUALIZATION")
    print(f"{'='*80}")
    print(f"Node:            {node}")
    print(f"Hydro Model:     {hydro_model_source}")
    print(f"Period:          {ssp_period}")
    print(f"Weight scheme:   {weight_scheme}")
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

    print(f"Loaded {len(filtered_datasets)} GCM scenarios for comparison")

    # Load monthly means for weight calculations
    monthly_means_file = f'./stats/datasets_{node}_monthly_means.csv'
    monthly_means_df = pd.read_csv(monthly_means_file, index_col=0)
    print(f"Loaded monthly means from: {monthly_means_file}")

    # Load weighted averages to identify which scenarios passed IQR filtering
    weighted_avgs_file = f'{stats_dir}/{node}_all_weighted_averages_{hydro_model_source}_{ssp_period}.csv'
    weighted_avgs_df = pd.read_csv(weighted_avgs_file, index_col=0)
    filtered_scenario_names = weighted_avgs_df.index.tolist()

    # all_scenarios_unfiltered_df = all scenarios before IQR filtering
    all_scenarios_unfiltered_df = monthly_prc_change[filtered_datasets].copy()

    # all_scenarios_filtered_df = only scenarios that passed IQR filter
    # (scenarios present in the weighted averages file)
    filtered_cols = [c for c in filtered_datasets if c in filtered_scenario_names]
    all_scenarios_filtered_df = monthly_prc_change[filtered_cols].copy()

    print(f"Full ensemble (pre-IQR): {len(all_scenarios_unfiltered_df.columns)} scenarios")
    print(f"Filtered ensemble (post-IQR): {len(all_scenarios_filtered_df.columns)} scenarios")

    # Create comprehensive synthesis figure
    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATION")
    print(f"{'='*80}\n")

    plot_selected_scenarios(
        selected_scenarios_df,
        all_scenarios_filtered_df,
        all_scenarios_unfiltered_df,
        monthly_means_df,
        node,
        hydro_model_source,
        ssp_period,
        figures_dir,
        weight_scheme=weight_scheme,
        scenarios_to_show=['low', 'high']
    )

    print(f"\n{'='*80}")
    print(f"VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nFigure saved to: {figures_dir}")


if __name__ == "__main__":
    main()
