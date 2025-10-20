import pandas as pd

fdir = './stats/diff_relative_to_dataset_baseline'
node = 'nyc_inflow'
monthly_means_prc_change = pd.read_csv(f'{fdir}/{node}_monthly_mean_prc_change_by_dataset_ssp_and_period.csv')


for ssp in ['ssp245', 'ssp370']:
    for period in ['2020_2059']:
        cols = [col for col in monthly_means_prc_change.columns if ssp in col]
        cols = [col for col in cols if period in col]
        cols = [col for col in cols if 'VIC' in col]
        print(f'Processing {ssp} {period} with datasets: {cols}')
        
        df_subset = monthly_means_prc_change[cols]
        
        # get the median, min and max
        median = df_subset.median(axis=1)
        min_val = df_subset.min(axis=1)
        max_val = df_subset.max(axis=1)
        
        # Save to csv
        output_df = pd.DataFrame({
            'month': monthly_means_prc_change.index,
            'median': median,
            'min': min_val,
            'max': max_val
        })
        output_df.to_csv(f'stats/summary_{node}_monthly_mean_prc_change_{ssp}_{period}.csv', index=False)