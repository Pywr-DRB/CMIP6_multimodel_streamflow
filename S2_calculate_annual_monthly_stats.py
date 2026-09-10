"""
Monthly and annual flow statistics for every dataset, plus percent changes relative to a baseline.

Loads daily flows for all datasets in pywrdrb/inputs and the pywrdrb built-in reconstructions,
computes NYC aggregate inflow (sum of the three reservoir inflow gages), and writes:
  stats/datasets_<node>_{annual_median,annual_means,annual_stds,monthly_means,monthly_stds}.csv
  stats/diff_relative_to_dataset_baseline/<node>_monthly_mean_{diff,frac,prc_change}_by_dataset_ssp_and_period.csv
  stats/diff_relative_to_reconstruction/<node>_monthly_mean_{diff,frac,prc_change}_by_dataset_ssp_and_period.csv

"dataset_baseline" compares each projection with the historic run of the same hydrologic model
and forcing (see utils.get_dataset_baseline). "reconstruction" compares everything with
pub_nhmv10_BC_withObsScaled.
"""
import os
import numpy as np
import pandas as pd
import pywrdrb
from config import DATASET_NAMES, INPUT_DIR, STATS_DIR
from utils import dataset_baselines

CONSIDER_NODES = ['nyc_inflow']
NYC_INFLOW_GAGES = ["01425000", "01417000", "01436000"]
RECONSTRUCTION = 'pub_nhmv10_BC_withObsScaled'


if __name__ == "__main__":
    # Register local datasets with pywrdrb
    pn_config = pywrdrb.get_pn_config()
    for dataset in DATASET_NAMES:
        pn_config[f"flows/{dataset}"] = os.path.join(INPUT_DIR, dataset)
    pywrdrb.load_pn_config(pn_config)

    # Every registered flow type: local datasets plus pywrdrb built-ins (skip reversed 'rev_' sets).
    # Sorted so CSV column order does not depend on the file system.
    pn = pywrdrb.get_pn_object()
    flowtypes = [k.replace("flows/", "") for k in pn.sc.to_dict().keys()]
    flowtypes = sorted(ft for ft in flowtypes if not ft.startswith('rev_'))

    data = pywrdrb.Data(results_sets=['major_flow'])
    data.load_hydrologic_model_flow(flowtypes)
    loaded_datasets = list(data.major_flow.keys())
    print(f"Loaded flows for {len(loaded_datasets)} datasets.")

    for dataset in loaded_datasets:
        df = data.major_flow[dataset][0]
        df['nyc_inflow'] = df[NYC_INFLOW_GAGES].sum(axis=1)

    for node in CONSIDER_NODES:
        node_flows = pd.DataFrame({d: data.major_flow[d][0][node].copy() for d in loaded_datasets})
        node_flows.index = pd.to_datetime(node_flows.index)

        monthly = node_flows.groupby([node_flows.index.year, node_flows.index.month]).sum()
        monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=['year', 'month'])
        annual = node_flows.groupby(node_flows.index.year).sum()

        # Periods with no data sum to exactly 0.0; treat those as missing
        monthly.replace(0.0, np.nan, inplace=True)
        annual.replace(0.0, np.nan, inplace=True)

        annual_median = pd.Series(np.nanmedian(annual, axis=0), index=annual.columns, name='median')
        annual_means = annual.mean()
        annual_stds = annual.std()
        monthly_means = monthly.groupby('month').mean()
        monthly_stds = monthly.groupby('month').std()

        os.makedirs(STATS_DIR, exist_ok=True)
        annual_median.to_csv(f'{STATS_DIR}/datasets_{node}_annual_median.csv')
        annual_means.to_csv(f'{STATS_DIR}/datasets_{node}_annual_means.csv')
        annual_stds.to_csv(f'{STATS_DIR}/datasets_{node}_annual_stds.csv')
        monthly_means.to_csv(f'{STATS_DIR}/datasets_{node}_monthly_means.csv')
        monthly_stds.to_csv(f'{STATS_DIR}/datasets_{node}_monthly_stds.csv')

        # Differences in monthly means relative to a baseline
        for use_dataset_baseline in [True, False]:
            monthly_means_diff = pd.DataFrame(index=monthly_means.index, columns=monthly_means.columns)
            monthly_means_frac = pd.DataFrame(index=monthly_means.index, columns=monthly_means.columns)
            monthly_means_prc_change = pd.DataFrame(index=monthly_means.index, columns=monthly_means.columns)

            for dataset in monthly_means.columns:
                if use_dataset_baseline:
                    baseline = dataset_baselines.get(dataset, RECONSTRUCTION)
                else:
                    baseline = RECONSTRUCTION

                if baseline not in monthly_means.columns:
                    print(f"Warning: Baseline '{baseline}' not found for dataset '{dataset}'. Skipping.")
                    continue

                baseline_means = monthly_means[baseline]
                monthly_means_diff[dataset] = monthly_means[dataset] - baseline_means
                monthly_means_frac[dataset] = monthly_means[dataset] / baseline_means
                monthly_means_prc_change[dataset] = (monthly_means_diff[dataset] / baseline_means) * 100

            subdir = 'diff_relative_to_dataset_baseline' if use_dataset_baseline else 'diff_relative_to_reconstruction'
            fdir = os.path.join(STATS_DIR, subdir)
            os.makedirs(fdir, exist_ok=True)
            monthly_means_diff.to_csv(f'{fdir}/{node}_monthly_mean_diff_by_dataset_ssp_and_period.csv')
            monthly_means_frac.to_csv(f'{fdir}/{node}_monthly_mean_frac_by_dataset_ssp_and_period.csv')
            monthly_means_prc_change.to_csv(f'{fdir}/{node}_monthly_mean_prc_change_by_dataset_ssp_and_period.csv')
            print(f"Saved monthly mean differences to {fdir}")
