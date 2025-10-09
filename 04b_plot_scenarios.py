import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm

from config import DATASET_NAMES
from utils import parse_dataset_settings_from_name, dataset_settings, dataset_baselines, calculate_iqr_scenarios
from styles import colordict_by_gcm, markerdict_by_hydrology_model

from config import DATASET_NAMES, DATASET_COLORS_BY_PERIOD
from plotting_functions import plot_monthly_stat_lines, plot_quantile_space_visualization

# pywrdrb_nodes list or subset of nodes to plot
CONSIDER_NODES = [
    'delMontague',
    'nyc_inflow'
]

plot_datasets = [d for d in DATASET_NAMES if 'VIC' in d]


def create_quantile_matrix(df_reordered, outlier_threshold=1.5):
    """
    Create a matrix of quantiles (12 months × 100 quantiles) from IQR-filtered data.
    
    Parameters:
    -----------
    df_reordered : pd.DataFrame
        13 rows (Jun-Jun) × N columns (GCM scenarios)
    outlier_threshold : float
        IQR multiplier for outlier removal
        
    Returns:
    --------
    np.ndarray of shape (12, 100) containing quantile values for each month
    """
    quantile_matrix = np.zeros((12, 100))
    
    # Process first 12 months (exclude duplicate June at index 12)
    for i in range(12):
        month_data = df_reordered.iloc[i].values
        
        # Remove outliers using IQR method
        q1 = np.percentile(month_data, 25)
        q3 = np.percentile(month_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - outlier_threshold * iqr
        upper_bound = q3 + outlier_threshold * iqr
        
        # Filter data
        filtered_data = month_data[(month_data >= lower_bound) & (month_data <= upper_bound)]
        
        # Calculate quantiles 1-100
        quantile_matrix[i, :] = np.percentile(filtered_data, range(1, 101))
    
    return quantile_matrix


def create_quantile_interpolated_scenario(quantile_matrix, jun_quantile, dec_quantile):
    """
    Create a scenario by linearly interpolating through quantile space between anchor months.
    
    Parameters:
    -----------
    quantile_matrix : np.ndarray
        Shape (12, 100) - quantile values for each month
    jun_quantile : int
        Quantile level (1-100) for June anchor
    dec_quantile : int
        Quantile level (1-100) for December anchor
        
    Returns:
    --------
    np.ndarray of length 13 (Jun-Jun) with interpolated values
    """
    scenario = np.zeros(13)
    
    # Anchor indices in reordered data: Jun=0, Dec=6
    jun_idx = 0
    dec_idx = 6
    
    for month_idx in range(13):
        if month_idx <= 6:  # Jun through Dec
            # Linear interpolation from June to December
            weight = month_idx / 6
            q = jun_quantile + weight * (dec_quantile - jun_quantile)
        else:  # Jan through Jun
            # Linear interpolation from December back to June
            weight = (month_idx - 6) / 6
            q = dec_quantile + weight * (jun_quantile - dec_quantile)
        
        # Lookup value at this quantile for this month
        # Handle duplicate June at end
        month_matrix_idx = month_idx if month_idx < 12 else 0
        q_idx = int(np.clip(q, 1, 100)) - 1  # Convert to 0-indexed
        
        scenario[month_idx] = quantile_matrix[month_matrix_idx, q_idx]
    
    return scenario




if __name__ == "__main__":
    
    node = 'nyc_inflow'
    use_dataset_baseline = True
    fdir = './stats/diff_relative_to_dataset_baseline' if use_dataset_baseline else './stats/diff_relative_to_reconstruction'
    monthly_means_prc_change = pd.read_csv(f'{fdir}/{node}_monthly_mean_prc_change_by_dataset_ssp_and_period.csv')

    # Filter datasets for BOTH SSPs (VIC, near-term 2020)
    filtered_datasets = [d for d in monthly_means_prc_change.columns 
                        if 'VIC' in d and ('ssp245' in d or 'ssp370' in d) and '2020' in d]
    
    df_combined_ssps = monthly_means_prc_change[filtered_datasets]
    
    # Reorder: Jun(6) through Dec(12), then Jan(1) through Jun(6)
    df_part1 = df_combined_ssps.loc[6:12]  # Jun through Dec
    df_part2 = df_combined_ssps.loc[1:6]   # Jan through Jun
    df_reordered = pd.concat([df_part1, df_part2])
    
    # Calculate full range (min/max) for background fill - no outlier removal
    full_min = df_reordered.min(axis=1).values
    full_max = df_reordered.max(axis=1).values
    
    # Create quantile matrix (12 months × 100 quantiles) with IQR filtering
    quantile_matrix = create_quantile_matrix(df_reordered, outlier_threshold=1.5)
    
    # Define scenarios as (June_quantile, December_quantile) pairs
    scenario_definitions = {
        'Jun_Low_Dec_Low': (10, 10),
        'Jun_Low_Dec_Med': (10, 50),
        'Jun_Low_Dec_High': (10, 90),
        'Jun_Med_Dec_Low': (50, 10),
        'Jun_Med_Dec_Med': (50, 50),
        'Jun_Med_Dec_High': (50, 90),
        'Jun_High_Dec_Low': (90, 10),
        'Jun_High_Dec_Med': (90, 50),
        'Jun_High_Dec_High': (90, 90),
    }
    
    # Generate all scenarios
    scenarios_dict = {}
    for scenario_name, (jun_q, dec_q) in scenario_definitions.items():
        scenarios_dict[scenario_name] = create_quantile_interpolated_scenario(
            quantile_matrix, jun_q, dec_q
        )
    
    # Convert to DataFrame with proper index (13 rows)
    month_index = list(range(6, 13)) + list(range(1, 7))  # [6,7,8,9,10,11,12,1,2,3,4,5,6]
    combined_scenarios_df = pd.DataFrame(scenarios_dict, index=month_index)
    
    # Save combined scenarios to CSV
    output_dir = f'{fdir}/seasonal_scenarios'
    
    os.makedirs(output_dir, exist_ok=True)
    combined_scenarios_df.to_csv(f'{output_dir}/{node}_quantile_interpolated_scenarios.csv')
    
    # Create quantile space visualization
    plot_quantile_space_visualization(quantile_matrix, scenario_definitions, node, 
                                     'figures/diff_relative_to_dataset_baseline')
    
    # Create figure with scenario time series
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot full range as background
    ax.fill_between(range(len(df_reordered)), 
                    full_min, 
                    full_max,
                    color='gray', 
                    alpha=0.15,
                    label='Full SSP range (all GCMs)')
    
    # Define colors and line styles for the 9 scenarios
    # Color by June level, linestyle by December level
    jun_colors = {
        'Low': '#2ca02c',    # green
        'Med': '#ff7f0e',    # orange
        'High': '#d62728',   # red
    }
    
    dec_linestyles = {
        'Low': ':',      # dotted
        'Med': '--',     # dashed
        'High': '-',     # solid
    }
    
    # Plot each scenario
    for col in combined_scenarios_df.columns:
        # Parse scenario name: 'Jun_Low_Dec_High'
        parts = col.split('_')
        jun_level = parts[1]   # Low/Med/High
        dec_level = parts[3]   # Low/Med/High
        
        color = jun_colors[jun_level]
        linestyle = dec_linestyles[dec_level]
        
        ax.plot(range(len(combined_scenarios_df)), 
                combined_scenarios_df[col].values,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                alpha=0.85,
                label=f'Jun: {jun_level}, Dec: {dec_level}')
    
    # Add reference line at zero
    ax.axhline(y=0.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Change in Monthly Mean Flow (%)\nRelative to Dataset Baseline', fontsize=12)
    ax.set_title(f'{node.replace("_", " ").title()} - Quantile-Interpolated Scenarios (2020-2059)\nSSP245 + SSP370', 
                 fontsize=14)
    
    # Set x-axis to show month names starting from June, ending with June
    month_labels = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 
                    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    ax.set_xticks(range(13))
    ax.set_xticklabels(month_labels)
    
    # Adjust y-limits with padding
    all_vals = df_reordered.values.flatten()
    ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
    ax.set_ylim(ymin * 1.1 if ymin < 0 else ymin * 0.9,
                ymax * 1.1 if ymax > 0 else ymax * 0.9)
    
    # Legend - outside plot to avoid clutter
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, ncol=1)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f'figures/diff_relative_to_dataset_baseline/{node}_quantile_interpolated_scenarios.png', 
                dpi=300, bbox_inches='tight')
    plt.show()