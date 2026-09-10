"""
Dataset-name parsing and baseline lookup shared by the P* and S* scripts.
"""
import os
import glob
from config import DATASET_NAMES, HUC_CODES, NETCDF_DIR


def parse_dataset_settings_from_name(dataset):
    """
    Split a dataset name into its parts. Names take one of three forms:
      <model>_RAPID_<gcm>_<ssp>_<run_id>_<downscaling>_<forcing>_<start>_<end>   projection
      <model>_RAPID_<forcing>_<start>_<end>                                       historic
      <model>_RAPID_<forcing>_<forcing_version>_<start>_<end>                     historic (versioned forcing)
    """
    parts = dataset.split('_')

    if len(parts) == 5:
        settings = {
            'hydrology_model': parts[0],
            'gcm': None,
            'ssp': None,
            'run_id': None,
            'downscaling_method': None,
            'forcing': parts[2],
            'start_year': int(parts[3]),
            'end_year': int(parts[4]),
        }
    elif len(parts) == 6:
        settings = {
            'hydrology_model': parts[0],
            'gcm': None,
            'ssp': None,
            'run_id': None,
            'downscaling_method': None,
            'forcing': parts[2],
            'start_year': int(parts[4]),
            'end_year': int(parts[5]),
        }
    elif len(parts) == 9:
        settings = {
            'hydrology_model': parts[0],
            'gcm': parts[2],
            'ssp': parts[3],
            'run_id': parts[4],
            'downscaling_method': parts[5],
            'forcing': parts[6],
            'start_year': int(parts[7]),
            'end_year': int(parts[8]),
        }
    else:
        raise ValueError(f"Dataset name '{dataset}' does not match expected format.")

    return settings


def verify_dataset_has_necessary_files(dataset):
    """Return True if NETCDF_DIR/<dataset>/ holds a netCDF file for every HUC code."""
    for huc_code in HUC_CODES:
        pattern = os.path.join(NETCDF_DIR, dataset, f"*{huc_code}*.nc")
        if not glob.glob(pattern):
            print(f"Missing file for HUC {huc_code} in dataset {dataset}.")
            return False
    return True


def get_dataset_baseline(dataset):
    """Name of the historic run that serves as baseline for `dataset` (same model and forcing)."""
    settings = parse_dataset_settings_from_name(dataset)

    if settings['hydrology_model'] == 'PRMS':
        if 'Livneh' in settings['forcing']:
            return 'PRMS_RAPID_Livneh2018_1950_2013'
        elif 'Daymet' in settings['forcing']:
            return 'PRMS_RAPID_Daymet2019_1980_2019'
        raise ValueError(f"Unknown forcing for PRMS hydrology model: {settings['forcing']}")

    elif settings['hydrology_model'] == 'VIC5':
        if 'Livneh' in settings['forcing']:
            return 'VIC5_RAPID_Livneh2018_v20200704L_1950_2013'
        elif 'Daymet' in settings['forcing']:
            return 'VIC5_RAPID_Daymet2019_v20200704D_1980_2019'
        raise ValueError(f"Unknown forcing for VIC5 hydrology model: {settings['forcing']}")

    return None


# dataset name -> baseline dataset name, for every dataset in pywrdrb/inputs
dataset_baselines = {}
for dataset in DATASET_NAMES:
    try:
        dataset_baselines[dataset] = get_dataset_baseline(dataset)
    except ValueError as e:
        print(f"Error getting baseline for dataset {dataset}: {e}")
        dataset_baselines[dataset] = None
