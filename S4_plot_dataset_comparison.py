import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

from config import DATASET_NAMES
from utils import parse_dataset_settings_from_name, dataset_settings, dataset_baselines, calculate_iqr_scenarios
from styles import colordict_by_gcm, markerdict_by_hydrology_model

from config import DATASET_NAMES, DATASET_COLORS_BY_PERIOD
from plotting_functions import plot_monthly_stat_lines

# pywrdrb_nodes list or subset of nodes to plot
CONSIDER_NODES = [
    'nyc_inflow'
]


plot_datasets = [d for d in DATASET_NAMES if 'VIC' in d]


if __name__ == "__main__":

    for node in CONSIDER_NODES:
        if node == 'delTrenton':
            continue
        
        monthly_means = pd.read_csv(f'stats/datasets_{node}_monthly_means.csv', index_col=0)
        
        loaded_datasets = monthly_means.columns.tolist()
        dataset_types = {}
        dataset_types['historic'] = [d for d in loaded_datasets if ('2059' not in d) and ('2060' not in d)]
        dataset_types['VIC_ssp126_hist'] = [d for d in loaded_datasets if ('VIC' in d) and ('ssp126' in d) and ('1980' in d)] 
        dataset_types['VIC_ssp245_near'] = [d for d in loaded_datasets if ('VIC' in d) and ('ssp245' in d) and ('2020' in d)]
        dataset_types['VIC_ssp245_far'] = [d for d in loaded_datasets if ('VIC' in d) and ('ssp245' in d) and ('2060' in d)]
        dataset_types['VIC_ssp370_near'] = [d for d in loaded_datasets if ('VIC' in d) and ('ssp370' in d) and ('2020' in d)]
        dataset_types['VIC_ssp370_far'] = [d for d in loaded_datasets if ('VIC' in d) and ('ssp370' in d) and ('2060' in d)]

        historic_datasets = dataset_types['historic']
        
        
        ## Plot annual mean flow distributions
        annual_means = pd.read_csv(f'stats/datasets_{node}_annual_means.csv', index_col=0)
        
        annual_means['dataset_type'] = annual_means.index.map(
            lambda x: next((dt for dt, ds in dataset_types.items() if x in ds), 'unknown')
        )
        
        annual_means['dataset_name'] = annual_means.index
        annual_means['annual_flow'] = annual_means.loc[:, '0']
        annual_means = annual_means.reset_index(drop=True)
        
        # print all dataset names with dataset_type == unknown
        for d in annual_means['dataset_name']:
            if annual_means.loc[annual_means['dataset_name'] == d, 'dataset_type'].values[0] == 'unknown':
                print(f"Dataset {d} has unknown dataset_type. Please check the dataset settings.")
                
        print(f'annual_means: {annual_means.shape}')
        print(f"annual_means columns: {annual_means.columns.tolist()}")
        print(f"annual_means index: {annual_means.index.tolist()}")
        
        annual_means['log_annual_flow'] = np.log10(annual_means['annual_flow']) 
             
        # Plot barplots of annual means by dataset type

        annual_means_subset = annual_means[annual_means['dataset_type'].isin(['VIC_ssp245_near', 'VIC_ssp370_near', 'VIC_ssp126_hist'])]
        
        # calculate percentage difference between pub_nhmv10_BC_withObsScaled and each other dataset
        ref_value = annual_means.loc[annual_means['dataset_name'] == 'pub_nhmv10_BC_withObsScaled', 'annual_flow'].values[0]
        annual_means_subset['pct_diff_from_reconstruction'] = (annual_means_subset['annual_flow'] - ref_value) / ref_value * 100.0
        
        # Model colors by dataset type
        dataset_type_colors = {
            'historic': 'blue',
            'VIC_ssp245_near': 'orange',
            'VIC_ssp370_near': 'maroon',
            'VIC_ssp126_hist': 'blue',
            'unknown': 'lightgrey'
        }
        
        
        # print the number of datasets in each dataset_type
        print(annual_means_subset['dataset_type'].value_counts())
        
        fig, ax = plt.subplots(figsize=(6,6))
        sns.kdeplot(data=annual_means_subset, 
                    x='pct_diff_from_reconstruction', 
                    hue='dataset_type',
                    alpha=0.2,
                    palette=dataset_type_colors,
                    bw_adjust=0.8,
                    fill=True,
                    lw=2,
                    ax=ax)
        # add vertical line for pub_nhmv10_BC_withObsScaled
        xval = annual_means.loc[annual_means['dataset_name'] == 'pub_nhmv10_BC_withObsScaled', 'log_annual_flow'].values[0]
        ymax = ax.get_ylim()[1]
        ax.vlines(0.0, 0, ymax, color='black', linestyle='--', lw=0.5)
        
        ax.set_xlabel('Change in annual mean flow (%)')
        plt.tight_layout()
        plt.savefig(f'figures/annual_totals/{node}_annual_mean_flow_percent_difference_by_dataset_type.png')
        plt.close(fig)


        # Plot monthly percent change only for VIC + nearterm datasets (ssp245 and ssp370, 2020 start)
        # Filter datasets for this SSP and start year

        use_dataset_baseline = True
        fdir = './stats/diff_relative_to_dataset_baseline' if use_dataset_baseline else './stats/diff_relative_to_reconstruction'
        monthly_means_prc_change = pd.read_csv(f'{fdir}/{node}_monthly_mean_prc_change_by_dataset_ssp_and_period.csv')

        filtered_datasets = [d for d in monthly_means.columns if 'VIC' in d and ('ssp245' in d or 'ssp370' in d) and '2020' in d]
        df = monthly_means_prc_change[filtered_datasets]
        # Plot each dataset
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for ssp in ['ssp245', 'ssp370']:
            filtered_datasets = [d for d in df.columns if ssp in d]

            plot_monthly_stat_lines(
                df[filtered_datasets],
                ax=ax,
                labelby='gcm',
                fill_between=True,
                fill_color=dataset_type_colors[f'VIC_{ssp}_near'],
                colordict=colordict_by_gcm,
                markerdict=markerdict_by_hydrology_model,
                linestyledict={}
            )
        
        # Set x and y limits
        ax.set_xlim(df.index.min(), df.index.max())
        ax.set_ylim(df.min().min()*1.1, 
                    df.max().max() * 1.1)

        # add horizontal line if specified
        add_horz_at = 0.0
        if add_horz_at is not None:
            ax.axhline(y=add_horz_at, color='grey', linestyle='--')
        
        # Add x and y labels
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Difference in Monthly Mean Flow (%)\nRelative to Dataset Baseline', fontsize=12)
        
        # save
        plt.savefig(f'figures/diff_relative_to_dataset_baseline/{node}_monthly_mean_prc_change_VIC_ssp245_and_ssp370_nearterm.png')
    
    # Modified section for plotting with IQR-based scenarios
    # This replaces the existing plotting section at the bottom of your script



    # Replace the existing plotting code at the bottom with this:
    use_dataset_baseline = True
    fdir = './stats/diff_relative_to_dataset_baseline' if use_dataset_baseline else './stats/diff_relative_to_reconstruction'
    monthly_means_prc_change = pd.read_csv(f'{fdir}/{node}_monthly_mean_prc_change_by_dataset_ssp_and_period.csv')

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color and style settings for scenarios
    scenario_styles = {
        'ssp245': {
            'color': 'orange',
            'low': {'linestyle': ':', 'alpha': 0.8, 'linewidth': 2},
            'medium': {'linestyle': '-', 'alpha': 1.0, 'linewidth': 2.5},
            'high': {'linestyle': '--', 'alpha': 0.8, 'linewidth': 2}
        },
        'ssp370': {
            'color': 'maroon', 
            'low': {'linestyle': ':', 'alpha': 0.8, 'linewidth': 2},
            'medium': {'linestyle': '-', 'alpha': 1.0, 'linewidth': 2.5},
            'high': {'linestyle': '--', 'alpha': 0.8, 'linewidth': 2}
        }
    }

    # Process each SSP
    for ssp in ['ssp245', 'ssp370']:
        # Filter datasets for this SSP (VIC, near-term 2020)
        filtered_datasets = [d for d in monthly_means_prc_change.columns 
                            if 'VIC' in d and ssp in d and '2020' in d]
        
        if filtered_datasets:
            df_ssp = monthly_means_prc_change[filtered_datasets]
            
            # Reorder: Jun(6) through Dec(12), then Jan(1) through Jun(6)
            # Need to handle both index reordering and appending June again
            df_part1 = df_ssp.loc[6:12]  # Jun through Dec
            df_part2 = df_ssp.loc[1:6]   # Jan through Jun
            df_ssp_reordered = pd.concat([df_part1, df_part2])
            
            # Calculate full range (min/max) for fill_between - no outlier removal
            full_min = df_ssp_reordered.min(axis=1).values
            full_max = df_ssp_reordered.max(axis=1).values
            
            # Fill between full range of all GCMs
            ax.fill_between(range(len(df_ssp_reordered)), 
                        full_min, 
                        full_max,
                        color=scenario_styles[ssp]['color'], 
                        alpha=0.15,
                        label=f'{ssp.upper()} full range')
            
            # Calculate IQR-based scenarios for the lines only
            scenarios = calculate_iqr_scenarios(df_ssp_reordered, outlier_threshold=1.5)
            
            # Plot individual scenario lines
            base_color = scenario_styles[ssp]['color']
            
            ax.plot(range(len(df_ssp_reordered)), scenarios['low'], 
                color=base_color,
                label=f'{ssp.upper()} low (10th %ile, outliers removed)',
                **scenario_styles[ssp]['low'])
            
            ax.plot(range(len(df_ssp_reordered)), scenarios['medium'],
                color=base_color,
                label=f'{ssp.upper()} median (outliers removed)',
                **scenario_styles[ssp]['medium'])
            
            ax.plot(range(len(df_ssp_reordered)), scenarios['high'],
                color=base_color,
                label=f'{ssp.upper()} high (90th %ile, outliers removed)',
                **scenario_styles[ssp]['high'])

    # Add reference line at zero
    ax.axhline(y=0.0, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

    # Set labels and limits
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Change in Monthly Mean Flow (%)\nRelative to Dataset Baseline', fontsize=12)
    ax.set_title(f'{node.replace("_", " ").title()} - Climate Scenarios (2020-2059)', fontsize=14)

    # Set x-axis to show month names starting from June, ending with June
    month_labels = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 
                    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    ax.set_xticks(range(13))
    ax.set_xticklabels(month_labels)

    # Adjust y-limits with some padding
    all_vals = []
    for ssp in ['ssp245', 'ssp370']:
        filtered_datasets = [d for d in monthly_means_prc_change.columns 
                            if 'VIC' in d and ssp in d and '2020' in d]
        if filtered_datasets:
            all_vals.extend(monthly_means_prc_change[filtered_datasets].values.flatten())

    if all_vals:
        ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
        ax.set_ylim(ymin * 1.1 if ymin < 0 else ymin * 0.9,
                    ymax * 1.1 if ymax > 0 else ymax * 0.9)

    # Legend
    ax.legend(loc='best', fontsize=9, ncol=2)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':')

    # Save figure
    plt.tight_layout()
    plt.savefig(f'figures/diff_relative_to_dataset_baseline/{node}_monthly_mean_prc_change_VIC_iqr_scenarios.png', dpi=300)
    plt.show()