"""
Extract daily streamflow at the Pywr-DRB nodes from the RAPID netCDF files in netcdf/
and save them as pywrdrb/inputs/<dataset>/gage_flow_mgd.csv.

Only needed when new netCDF datasets are added; the gage_flow_mgd.csv files are tracked
in git. Requires the raw downloads in netcdf/<dataset>/ (see README) and
data/drb_pywrdrb_node_metadata.csv, which maps each Pywr-DRB node to a NHDPlus COMID.
"""
import os
import numpy as np
import pandas as pd
import netCDF4 as nc
from pywrdrb.utils.constants import cfs_to_mgd

from config import HUC_CODES, NETCDF_DIR, INPUT_DIR, ROOT_DIR
from utils import verify_dataset_has_necessary_files


def extract_pywrdrb_from_model_netcdfs(dataset_files, node_metadata):
    """
    Read flows at every Pywr-DRB node found in `dataset_files` (one netCDF per HUC).
    Returns a DataFrame of daily flow in MGD indexed by datetime, one column per node.
    """
    flow_df = {}
    for fname in dataset_files:
        ds = nc.Dataset(fname)

        # Only a subset of COMIDs is present in each HUC file
        all_huc_comids = ds.variables['COMID'][:].astype(int)
        node_metadata_huc = node_metadata[node_metadata.index.isin(all_huc_comids)]

        for comid in node_metadata_huc.index:
            name = node_metadata_huc.loc[comid, 'name']
            ds_comid_idx = np.where(ds.variables['COMID'][:] == comid)[0]
            comid_flow = ds.variables['RAPID_dy_cfs'][ds_comid_idx, :].data.flatten()

            # Some COMIDs map to more than one node name
            if type(name) is pd.Series:
                for n in name:
                    flow_df[n] = comid_flow
            else:
                flow_df[name] = comid_flow

    datetime = pd.to_datetime(ds.variables['Time_dy'][:], format='%Y%m%d')
    flow_df = pd.DataFrame(flow_df, index=datetime) * cfs_to_mgd
    flow_df.index.name = 'datetime'
    return flow_df


if __name__ == "__main__":
    node_metadata = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'drb_pywrdrb_node_metadata.csv'))
    node_metadata.set_index('comid', inplace=True)

    if not os.path.isdir(NETCDF_DIR):
        raise FileNotFoundError(f"No netCDF downloads found in {NETCDF_DIR}")
    datasets = sorted(d for d in os.listdir(NETCDF_DIR) if os.path.isdir(os.path.join(NETCDF_DIR, d)))

    for dataset in datasets:
        output_file = os.path.join(INPUT_DIR, dataset, "gage_flow_mgd.csv")
        if os.path.exists(output_file):
            print(f"Skipping {dataset}, already processed.")
            continue
        if not verify_dataset_has_necessary_files(dataset):
            print(f"Skipping {dataset}, missing netCDF files.")
            continue

        dataset_dir = os.path.join(NETCDF_DIR, dataset)
        dataset_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
                         if any(huc in f for huc in HUC_CODES)]

        print(f"Processing {dataset}...")
        flow_df = extract_pywrdrb_from_model_netcdfs(dataset_files, node_metadata)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        flow_df.to_csv(output_file)
