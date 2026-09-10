"""
Plot aggregate NYC reservoir storage (Cannonsville + Pepacton + Neversink) from the
Pywr-DRB simulations of a set of datasets. Edit DATASETS to choose which runs to show.
Saves figures/nyc_reservoir_storages.png.
"""
import os
import matplotlib.pyplot as plt
import pywrdrb
from config import DATASET_NAMES, ROOT_DIR, FIGURES_DIR

# Default: near-term projections under SSP2-4.5 and SSP3-7.0. For the two Livneh historic runs use
# DATASETS = ['PRMS_RAPID_Livneh2018_1950_2013', 'VIC5_RAPID_Livneh2018_v20200704L_1950_2013']
DATASETS = [d for d in DATASET_NAMES if '2020_2059' in d and ('ssp245' in d or 'ssp370' in d)]
NYC_RESERVOIRS = ['cannonsville', 'pepacton', 'neversink']


if __name__ == "__main__":
    output_files = [os.path.join(ROOT_DIR, 'pywrdrb', 'outputs', f'{d}.hdf5') for d in DATASETS]
    data = pywrdrb.Data(results_sets=['res_storage'], print_status=True)
    data.load_output(output_filenames=output_files)

    fig, ax = plt.subplots(figsize=(12, 6))
    for dataset in DATASETS:
        storage = data.res_storage[dataset][0][NYC_RESERVOIRS].sum(axis=1)
        ax.plot(storage.index, storage, label=dataset, alpha=0.3)

    ax.set_title(f"NYC aggregate reservoir storage across {len(DATASETS)} datasets")
    ax.set_xlabel("Date")
    ax.set_ylabel("Storage (MG)")
    if len(DATASETS) <= 10:
        ax.legend(fontsize=8)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'nyc_reservoir_storages.png'), dpi=200)
