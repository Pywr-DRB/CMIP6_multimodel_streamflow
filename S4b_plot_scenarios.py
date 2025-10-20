import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from config import DATASET_NAMES
from plotting_functions import plot_monthly_stat_lines, plot_quantile_space_visualization, plot_quantile_space_with_gcm_traces

# pywrdrb_nodes list or subset of nodes to plot
CONSIDER_NODES = [
    'delMontague',
    'nyc_inflow'
]


def transform_to_quantile_space(df_reordered, quantile_matrix):
    """
    Transform streamflow shift values to quantile space.
    
    Parameters:
    -----------
    df_reordered : pd.DataFrame
        13 rows × N columns (GCM scenarios), with flow change values
    quantile_matrix : np.ndarray
        Shape (12, 100) - quantile values for each month
        
    Returns:
    --------
    pd.DataFrame with same shape as df_reordered, containing quantile values (0-100)
    """
    df_quantiles = pd.DataFrame(index=df_reordered.index, columns=df_reordered.columns, dtype=float)
    
    for i, month_idx in enumerate(df_reordered.index):
        # Get the appropriate month row from quantile_matrix
        # Handle duplicate June at end (index 12 maps to June = index 0)
        if i < 12:
            month_matrix_idx = i
        else:  # i == 12, duplicate June
            month_matrix_idx = 0  # June
        
        month_quantiles = quantile_matrix[month_matrix_idx, :]
        
        # For each dataset, find which quantile its value corresponds to
        for col in df_reordered.columns:
            value = df_reordered.loc[month_idx, col]
            
            # Find closest quantile (searchsorted finds insertion point)
            q_idx = np.searchsorted(month_quantiles, value)
            q_idx = np.clip(q_idx, 0, 99)  # Clip to valid range
            
            # Convert to 1-100 scale
            df_quantiles.loc[month_idx, col] = q_idx + 1
    
    return df_quantiles


def filter_gcm_scenarios_by_iqr(df_reordered, outlier_threshold=1.5):
    """
    Filter GCM scenarios by removing outliers using IQR method across all months.
    
    Parameters:
    -----------
    df_reordered : pd.DataFrame
        13 rows (Jun-Jun) × N columns (GCM scenarios), with flow change values
    outlier_threshold : float
        IQR multiplier for outlier removal
        
    Returns:
    --------
    pd.DataFrame: Subset of df_reordered with outlier scenarios removed
    """
    # Track which scenarios are outliers in any month
    outlier_scenarios = set()
    
    # Process first 12 months (exclude duplicate June at end)
    for i in range(12):
        month_data = df_reordered.iloc[i].values
        
        # Calculate IQR bounds for this month
        q1 = np.percentile(month_data, 25)
        q3 = np.percentile(month_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - outlier_threshold * iqr
        upper_bound = q3 + outlier_threshold * iqr
        
        # Find scenarios that are outliers in this month
        for j, scenario_name in enumerate(df_reordered.columns):
            value = month_data[j]
            if value < lower_bound or value > upper_bound:
                outlier_scenarios.add(scenario_name)
    
    # Return dataframe with outlier scenarios removed
    filtered_scenarios = [col for col in df_reordered.columns if col not in outlier_scenarios]
    df_filtered = df_reordered[filtered_scenarios].copy()
    
    print(f"IQR Filtering: Removed {len(outlier_scenarios)} outlier scenarios out of {len(df_reordered.columns)}")
    print(f"Remaining scenarios for analysis: {len(filtered_scenarios)}")
    
    return df_filtered


def create_quantile_matrix(df_filtered):
    """
    Create a matrix of quantiles (12 months × 100 quantiles) from pre-filtered data.
    
    Parameters:
    -----------
    df_filtered : pd.DataFrame
        13 rows (Jun-Jun) × N columns (pre-filtered GCM scenarios)
        
    Returns:
    --------
    np.ndarray of shape (12, 100) containing quantile values for each month
    """
    quantile_matrix = np.zeros((12, 100))
    
    # Process first 12 months (exclude duplicate June at index 12)
    for i in range(12):
        month_data = df_filtered.iloc[i].values
        
        # Calculate quantiles 1-100 (no additional filtering needed - already done)
        quantile_matrix[i, :] = np.percentile(month_data, range(1, 101))
    
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


def find_closest_gcm_scenarios(df_reordered_quantiles, scenario_definitions, quantile_matrix):
    """
    Find GCM scenarios with minimum weighted Euclidean distance to anchor-based scenarios in quantile space.
    Uses weighted distance that emphasizes June and December anchor periods.
    
    Parameters:
    -----------
    df_reordered_quantiles : pd.DataFrame
        GCM scenarios in quantile space (13 months × N GCMs)
    scenario_definitions : dict
        Anchor scenarios as (June_quantile, December_quantile) pairs
    quantile_matrix : np.ndarray
        Shape (12, 100) - quantile values for each month
        
    Returns:
    --------
    dict: Maps scenario names to closest GCM info {'gcm_name': str, 'distance': float, 'quantiles': np.array}
    """
    closest_gcms = {}
    
    # Define weight vector for 13 months (Jun-Jun reordered: [6,7,8,9,10,11,12,1,2,3,4,5,6])
    # Index mapping: Jun=0, Jul=1, Aug=2, Sep=3, Oct=4, Nov=5, Dec=6, Jan=7, Feb=8, Mar=9, Apr=10, May=11, Jun=12
    weights = np.zeros(13)
    
    # High weights for anchor months
    weights[0] = 0.3   # June (primary anchor)
    weights[6] = 0.3   # December (primary anchor)
    weights[12] = 0.3  # June duplicate (same as June)
    
    # Medium weights for adjacent months
    weights[1] = 0.3   # July (June + 1)
    weights[11] = 0.3  # May (June - 1)
    weights[5] = 0.3   # November (December - 1)
    weights[7] = 0.3   # January (December + 1)
    
    # Low weights for secondary months
    weights[2] = 0.0   # August (June + 2)
    weights[10] = 0.0  # April (June - 2)
    weights[4] = 0.0   # October (December - 2)
    weights[8] = 0.0   # February (December + 2)
    
    # Zero weights for all other months (March, September)
    weights[3] = 0.0   # September
    weights[9] = 0.0   # March
    
    # Normalize weights to sum to 1
    weights /= np.sum(weights)
    
    print(f"Weight vector (Jun-Jun): {weights}")
    print(f"Month order: ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']")
    
    # Generate quantile trajectories for each anchor scenario
    anchor_quantiles = {}
    for scenario_name, (jun_q, dec_q) in scenario_definitions.items():
        # Create the idealized quantile trajectory (13 months)
        quantile_trajectory = np.zeros(13)
        
        for month_idx in range(13):
            if month_idx <= 6:  # Jun through Dec
                weight_interp = month_idx / 6
                q = jun_q + weight_interp * (dec_q - jun_q)
            else:  # Jan through Jun (wrapping back)
                weight_interp = (month_idx - 6) / 6
                q = dec_q + weight_interp * (jun_q - dec_q)
            
            quantile_trajectory[month_idx] = q
        
        anchor_quantiles[scenario_name] = quantile_trajectory
    
    # Find closest GCM to each anchor scenario using weighted distance
    for scenario_name, target_quantiles in anchor_quantiles.items():
        min_distance = float('inf')
        closest_gcm = None
        
        # Calculate weighted distance to each GCM trajectory
        for gcm_col in df_reordered_quantiles.columns:
            gcm_quantiles = df_reordered_quantiles[gcm_col].values
            
            # Ensure both arrays have same length (13)
            if len(gcm_quantiles) != len(target_quantiles):
                continue
            
            # Weighted Euclidean distance in quantile space
            squared_diffs = (gcm_quantiles - target_quantiles) ** 2
            weighted_squared_diffs = weights * squared_diffs
            distance = np.sqrt(np.sum(weighted_squared_diffs))
            
            if distance < min_distance:
                min_distance = distance
                closest_gcm = gcm_col
        
        closest_gcms[scenario_name] = {
            'gcm_name': closest_gcm,
            'distance': min_distance,
            'quantiles': df_reordered_quantiles[closest_gcm].values,
            'target_quantiles': target_quantiles
        }
    
    return closest_gcms


def create_gcm_derived_scenarios_df(closest_gcms, df_reordered, month_index):
    """
    Create DataFrame of GCM-derived scenarios using the actual flow values.
    
    Parameters:
    -----------
    closest_gcms : dict
        Output from find_closest_gcm_scenarios
    df_reordered : pd.DataFrame
        Original flow change data (13 months × N GCMs)
    month_index : list
        Month indices for the DataFrame
        
    Returns:
    --------
    pd.DataFrame: GCM-derived scenarios with actual flow values
    """
    gcm_scenarios_dict = {}
    
    for scenario_name, gcm_info in closest_gcms.items():
        gcm_name = gcm_info['gcm_name']
        gcm_scenarios_dict[scenario_name] = df_reordered[gcm_name].values
    
    return pd.DataFrame(gcm_scenarios_dict, index=month_index)


def plot_quantile_space_with_gcm_derived_scenarios(quantile_matrix, gcm_derived_scenarios_df, closest_gcms, node, fname):
    """
    Create a quantile space heatmap with GCM-derived scenario trajectories overlaid.
    
    Parameters:
    -----------
    quantile_matrix : np.ndarray
        Shape (12, 100) - quantile values for each month
    gcm_derived_scenarios_df : pd.DataFrame
        GCM-derived scenarios with actual flow values (13 months × N scenarios)
    closest_gcms : dict
        Information about which GCM was selected for each scenario
    node : str
        Node name for labeling
    fname : str
        Output filename for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap background showing flow change values
    # Use the quantile matrix transposed so quantiles are on y-axis, months on x-axis
    heatmap_data = quantile_matrix.T  # Shape: (100, 12)
    
    # Plot heatmap with diverging colormap centered at 0
    norm = TwoSlopeNorm(vmin=heatmap_data.min(), vcenter=0, vmax=heatmap_data.max())
    im = ax.imshow(heatmap_data, aspect='auto', origin='lower', 
                   cmap='RdBu', norm=norm,
                   interpolation='bilinear',
                   extent=[0, 12, 0, 100])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Flow Change (%)', pad=0.02)
    
    # Convert GCM scenarios to quantile space for plotting
    gcm_quantiles_dict = {}
    for scenario_name in gcm_derived_scenarios_df.columns:
        quantile_trajectory = []
        
        # Convert first 12 months to quantile space
        for month_idx in range(12):
            flow_value = gcm_derived_scenarios_df.iloc[month_idx][scenario_name]
            month_quantiles = quantile_matrix[month_idx, :]
            
            # Find which quantile this flow value corresponds to
            q_idx = np.searchsorted(month_quantiles, flow_value)
            q_idx = np.clip(q_idx, 0, 99)  # Clip to valid range
            quantile_level = q_idx + 1  # Convert to 1-100 scale
            
            quantile_trajectory.append(quantile_level)
        
        gcm_quantiles_dict[scenario_name] = quantile_trajectory
    
    # Define colors for different scenario groups (matching existing style)
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
    
    # Plot GCM-derived scenario paths
    for scenario_name, quantile_trajectory in gcm_quantiles_dict.items():
        # Parse scenario name for styling
        parts = scenario_name.split('_')
        jun_level = parts[1]   # Low/Med/High
        dec_level = parts[3]   # Low/Med/High
        
        color = jun_colors[jun_level]
        linestyle = dec_linestyles[dec_level]
        
        # Get GCM info
        gcm_name = closest_gcms[scenario_name]['gcm_name']
        distance = closest_gcms[scenario_name]['distance']
        
        # Create path through quantile space (months on x-axis, quantiles on y-axis)
        months_plot = np.arange(0, 12)
        
        # Plot the path
        ax.plot(months_plot, quantile_trajectory, 
                color=color, linestyle=linestyle, linewidth=2.5,
                alpha=0.9, zorder=10,
                label=f'Jun: {jun_level}, Dec: {dec_level} (d={distance:.1f})')
        
        # Add anchor points at June (month 0) and December (month 6)
        ax.scatter([0, 6], [quantile_trajectory[0], quantile_trajectory[6]], 
                  color=color, s=100, zorder=11, 
                  edgecolors='white', linewidths=1.5)
    
    # Customize axes (matching existing style)
    ax.set_xlabel('Month', fontsize=13, fontweight='bold')
    ax.set_ylabel('Quantile (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'{node.replace("_", " ").title()} - GCM-Derived Scenarios in Quantile Space\n' + 
                 'Selected Models Closest to Anchor-Based Scenarios',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis ticks and labels
    month_labels = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 
                    'Jan', 'Feb', 'Mar', 'Apr', 'May']
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels, fontsize=10)
    
    # Set y-axis ticks
    ax.set_yticks([0, 10, 25, 50, 75, 90, 100])
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 12)
    
    # Add grid (matching existing style)
    ax.grid(True, alpha=0.3, linestyle=':', color='white', linewidth=0.5, zorder=5)
    
    # Legend
    ax.legend(loc='upper left', fontsize=9, 
             framealpha=0.95, edgecolor='black')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')




if __name__ == "__main__":
    
    node = 'nyc_inflow'
    use_dataset_baseline = True
    hydro_model_source = 'VIC' # or 'PRMS'
    fdir = './stats/diff_relative_to_dataset_baseline' if use_dataset_baseline else './stats/diff_relative_to_reconstruction'
    monthly_means_prc_change = pd.read_csv(f'{fdir}/{node}_monthly_mean_prc_change_by_dataset_ssp_and_period.csv')

    # Filter datasets for BOTH SSPs (by hydro source, near-term 2020)
    filtered_datasets = [d for d in monthly_means_prc_change.columns 
                        if hydro_model_source in d and ('ssp245' in d or 'ssp370' in d) and '2020' in d]
    
    df_combined_ssps = monthly_means_prc_change[filtered_datasets]
    
    # Reorder: Jun(6) through Dec(12), then Jan(1) through Jun(6)
    df_part1 = df_combined_ssps.loc[6:12]  # Jun through Dec (7 months: 6,7,8,9,10,11,12)
    df_part2 = df_combined_ssps.loc[1:6]   # Jan through Jun (6 months: 1,2,3,4,5,6)
    
    # Add duplicate June at the end (create new row with index 13)
    june_duplicate = df_combined_ssps.loc[6:6].copy()  # Copy June row
    june_duplicate.index = [13]  # Give it a new index
    
    df_reordered = pd.concat([df_part1, df_part2, june_duplicate])
    
    # Calculate full range (min/max) for background fill - using all data
    full_min = df_reordered.min(axis=1).values
    full_max = df_reordered.max(axis=1).values
    
    # Apply IQR filtering to remove outlier scenarios
    df_filtered = filter_gcm_scenarios_by_iqr(df_reordered, outlier_threshold=1.5)
    
    # Create quantile matrix using only filtered scenarios
    quantile_matrix = create_quantile_matrix(df_filtered)

    # Transform filtered datasets to quantile space
    df_filtered_quantiles = transform_to_quantile_space(df_filtered, quantile_matrix)
    
    # Save quantile-space data
    output_dir = f'{fdir}/seasonal_scenarios'
    os.makedirs(output_dir, exist_ok=True)
    df_filtered_quantiles.to_csv(f'{output_dir}/{node}_gcm_quantile_traces_filtered.csv')


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


    combined_scenarios_df.to_csv(f'{output_dir}/{node}_quantile_interpolated_scenarios_{hydro_model_source}.csv')

    # Find closest GCM scenarios to anchor points (using only filtered data)
    closest_gcms = find_closest_gcm_scenarios(df_filtered_quantiles, scenario_definitions, quantile_matrix)
    
    # Create DataFrame with GCM-derived scenarios (actual flow values from filtered data)
    gcm_derived_scenarios_df = create_gcm_derived_scenarios_df(closest_gcms, df_filtered, month_index)
    
    # Save GCM-derived scenarios
    gcm_derived_scenarios_df.to_csv(f'{output_dir}/{node}_gcm_derived_scenarios_{hydro_model_source}.csv')

    # Print matching information
    print(f"\nClosest GCM matches for {node}:")
    print("-" * 50)
    for scenario_name, info in closest_gcms.items():
        print(f"{scenario_name}: {info['gcm_name']} (distance: {info['distance']:.2f})")
    
    # Create quantile space visualization
    fname = f'figures/diff_relative_to_dataset_baseline/quantile_space_visualization_{hydro_model_source}.png'
    plot_quantile_space_visualization(quantile_matrix, scenario_definitions, node, 
                                     fname=fname)

    fname = f'figures/diff_relative_to_dataset_baseline/quantile_space_gcm_traces_{hydro_model_source}.png'
    plot_quantile_space_with_gcm_traces(quantile_matrix, df_filtered_quantiles, node,
                                       fname=fname)
    
    # Create quantile space visualization with GCM-derived scenarios
    fname = f'figures/diff_relative_to_dataset_baseline/quantile_space_gcm_derived_scenarios_{hydro_model_source}.png'
    plot_quantile_space_with_gcm_derived_scenarios(quantile_matrix, gcm_derived_scenarios_df, closest_gcms, node, fname)

    # Create figure comparing interpolated vs GCM-derived scenarios
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
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
    
    month_labels = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 
                    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    
    # TOP PANEL: Quantile-interpolated scenarios
    ax1.fill_between(range(len(df_reordered)), 
                     full_min, 
                     full_max,
                     color='gray', 
                     alpha=0.15,
                     label='Full SSP range (all GCMs)')
    
    # Plot each interpolated scenario
    for col in combined_scenarios_df.columns:
        # Parse scenario name: 'Jun_Low_Dec_High'
        parts = col.split('_')
        jun_level = parts[1]   # Low/Med/High
        dec_level = parts[3]   # Low/Med/High
        
        color = jun_colors[jun_level]
        linestyle = dec_linestyles[dec_level]
        
        ax1.plot(range(len(combined_scenarios_df)), 
                 combined_scenarios_df[col].values,
                 color=color,
                 linestyle=linestyle,
                 linewidth=2.0,
                 alpha=0.85,
                 label=f'Jun: {jun_level}, Dec: {dec_level}')
    
    ax1.axhline(y=0.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
    ax1.set_ylabel('Change in Monthly Mean Flow (%)\nRelative to Dataset Baseline', fontsize=12)
    ax1.set_title(f'{node.replace("_", " ").title()} - Quantile-Interpolated Scenarios (2020-2059)', fontsize=14)
    ax1.set_xticks(range(13))
    ax1.set_xticklabels(month_labels)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, ncol=1)
    ax1.grid(True, alpha=0.3, linestyle=':')
    
    # BOTTOM PANEL: GCM-derived scenarios
    ax2.fill_between(range(len(df_reordered)), 
                     full_min, 
                     full_max,
                     color='gray', 
                     alpha=0.15,
                     label='Full SSP range (all GCMs)')
    
    # Plot each GCM-derived scenario
    for col in gcm_derived_scenarios_df.columns:
        # Parse scenario name: 'Jun_Low_Dec_High'
        parts = col.split('_')
        jun_level = parts[1]   # Low/Med/High
        dec_level = parts[3]   # Low/Med/High
        
        color = jun_colors[jun_level]
        linestyle = dec_linestyles[dec_level]
        
        # Get GCM name for this scenario
        gcm_name = closest_gcms[col]['gcm_name']
        distance = closest_gcms[col]['distance']
        
        ax2.plot(range(len(gcm_derived_scenarios_df)), 
                 gcm_derived_scenarios_df[col].values,
                 color=color,
                 linestyle=linestyle,
                 linewidth=2.0,
                 alpha=0.85,
                 label=f'Jun: {jun_level}, Dec: {dec_level} (d={distance:.1f})')
    
    ax2.axhline(y=0.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Change in Monthly Mean Flow (%)\nRelative to Dataset Baseline', fontsize=12)
    ax2.set_title(f'GCM-Derived Scenarios (Closest to Anchor Points)', fontsize=14)
    ax2.set_xticks(range(13))
    ax2.set_xticklabels(month_labels)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, ncol=1)
    ax2.grid(True, alpha=0.3, linestyle=':')
    
    # Adjust y-limits for both panels (using full data range)
    all_vals = df_reordered.values.flatten()
    ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
    ylim_min = ymin * 1.1 if ymin < 0 else ymin * 0.9
    ylim_max = ymax * 1.1 if ymax > 0 else ymax * 0.9
    ax1.set_ylim(ylim_min, ylim_max)
    ax2.set_ylim(ylim_min, ylim_max)
    
    # Save figure
    plt.tight_layout()
    fname = f'figures/diff_relative_to_dataset_baseline/{node}_scenario_comparison_{hydro_model_source}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')