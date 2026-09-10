"""
Build the Pywr-DRB inputs derived from gage_flow_mgd.csv for every dataset in pywrdrb/inputs:
catchment inflows (upstream flows subtracted, accounting for travel time), 1-4 day ahead
predicted inflows used for NYC release decisions, and extrapolated and predicted NYC and NJ
diversions. Datasets are split across MPI ranks:
    mpirun -n <N> python P2_prep_pywrdrb_inputs.py
"""
import os
import pandas as pd
from mpi4py import MPI

import pywrdrb
from pywrdrb.pre.flows import _subtract_upstream_catchment_inflows

from config import DATASET_NAMES, INPUT_DIR

# Set a stage to False to skip it when its outputs already exist
REDO_INFLOW_CALCULATION = True
REDO_INFLOW_PREDICTION = True
REDO_DIVERSION_EXTRAPOLATION = True
REDO_DIVERSION_PREDICTION = True


USE_MPI = True
if USE_MPI:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"MPI enabled: rank {rank} of {size}")
else:
    comm = None
    rank = 0
    size = 1
    print("MPI not enabled.")

# Register local datasets with pywrdrb
pn_config = pywrdrb.get_pn_config()
for dataset in DATASET_NAMES:
    pn_config[f"flows/{dataset}"] = os.path.join(INPUT_DIR, dataset)
pywrdrb.load_pn_config(pn_config)

if USE_MPI:
    local_rank_datasets = [
        dataset for i, dataset in enumerate(DATASET_NAMES) if i % size == rank
    ]
else:
    local_rank_datasets = DATASET_NAMES


if __name__ == "__main__":

    print(f"Rank {rank} preparing {len(local_rank_datasets)} datasets for Pywr-DRB...")

    for dataset in local_rank_datasets:

        ## Calculate catchment inflows
        if REDO_INFLOW_CALCULATION:
            f = os.path.join(INPUT_DIR, dataset, "gage_flow_mgd.csv")
            flow_df = pd.read_csv(f, index_col=0, parse_dates=True)

            # Iteratively subtract upstream catchment flows
            inflow_df = _subtract_upstream_catchment_inflows(flow_df)
            inflow_df.index.name = "datetime"
            
            f = os.path.join(INPUT_DIR, dataset, "catchment_inflow_mgd.csv")
            inflow_df.to_csv(f)

        ## Generate predicted inflows
        if REDO_INFLOW_PREDICTION:        
            print(f"Generating predicted inflows for {dataset}...")
            inflow_predictor = pywrdrb.pre.PredictedInflowPreprocessor(
                flow_type=dataset
            )
            
            # Predict and save 
            inflow_predictor.load()
            inflow_predictor.process()
            inflow_predictor.save()
        
        ### Generate extrapolated diversions
        if REDO_DIVERSION_EXTRAPOLATION:
            print(f"Generating extrapolated NYC diversions for {dataset}...")
            # NYC
            nyc_diversion_preprocessor = pywrdrb.pre.ExtrapolatedDiversionPreprocessor(
                loc="nyc",
                flow_type=dataset,
            )
            
            nyc_diversion_preprocessor.load()
            nyc_diversion_preprocessor.process()
            nyc_diversion_preprocessor.save()
            nyc_diversion_preprocessor.plot(kind="regressions")
            nyc_diversion_preprocessor.plot(kind="diversions")
            
            print(f"Generating extrapolated NJ diversions for {dataset}...")
            # NJ
            nj_diversion_preprocessor = pywrdrb.pre.ExtrapolatedDiversionPreprocessor(
                loc="nj",
                flow_type=dataset,
            )
            nj_diversion_preprocessor.load()
            nj_diversion_preprocessor.process()
            nj_diversion_preprocessor.save()
            nj_diversion_preprocessor.plot(kind="regressions")
            nj_diversion_preprocessor.plot(kind="diversions")
        
        ### Generate predicted diversions
        if REDO_DIVERSION_PREDICTION:
            print(f"Generating predicted diversions for {dataset}...")
            diversion_predictor = pywrdrb.pre.PredictedDiversionPreprocessor(
                flow_type=dataset,
            )
            diversion_predictor.load()
            diversion_predictor.process()
            diversion_predictor.save()
