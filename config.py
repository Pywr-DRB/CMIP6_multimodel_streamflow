"""
Shared settings for the Pywr-DRB (P*) and scenario-selection (S*) workflows.
"""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Raw RAPID netCDF downloads, read by P1 only (not tracked in git).
NETCDF_DIR = os.path.join(ROOT_DIR, 'netcdf')

# One folder per dataset holding gage_flow_mgd.csv and derived Pywr-DRB inputs.
# This folder is tracked in git and defines the dataset list.
INPUT_DIR = os.path.join(ROOT_DIR, 'pywrdrb', 'inputs')
if not os.path.exists(INPUT_DIR):
    raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

DATASET_NAMES = sorted(
    name for name in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, name)) and not name.startswith('.')
)

STATS_DIR = os.path.join(ROOT_DIR, 'stats')
FIGURES_DIR = os.path.join(ROOT_DIR, 'figures')

# HUC8 codes covering the Delaware River Basin; used to pick netCDF files in P1.
HUC_CODES = [
    '02040101N', '02040102N', '02040103N',
    '02040104N', '02040105N', '02040106N',
    '02040201N', '02040202N', '02040203N',
    '02040204N', '02040205N', '02040206N', '02040207N',
]

# Scenario selection settings shared by S3, S4 and S5.
NODE = 'nyc_inflow'
HYDRO_MODEL = 'PRMS'            # PRMS preferred over VIC5 based on S1
SSP_PERIOD = '2020_2059'
SSPS = ('ssp245', 'ssp370')
USE_DATASET_BASELINE = True     # % change relative to each dataset's own historic run
BASELINE_SUBDIR = ('diff_relative_to_dataset_baseline' if USE_DATASET_BASELINE
                   else 'diff_relative_to_reconstruction')

SCENARIO_LABELS = {'low': 'Wetter winter, drier summer', 'high': 'Wetter winter'}
SCENARIO_COLORS = {'low': '#ed9f1c', 'high': '#009e73'}
