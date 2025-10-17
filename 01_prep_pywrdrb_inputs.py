### Processing additional inputs ########################
# We need to create two CSV files from the gage flow:
# 1. catchment_inflow_mgd.csv : 
#       This contains the marginal inflow at each node, 
#       calculated by iteratively subtracting upstream flows,
#       and accounting for travel time. 
#       E.g., inflow at Montague is calculated by as total Montague flow
#       minus upstream reservoir inflows. 
# 2. predicted_inflow_mgd.csv:
#       This contains 1-4 day ahead inflow predictions made
#       using a AR model. These are used in pywrdrb to determine
#       NYC releases for Montague, accounting for travel time


import os
import pandas as pd
import pywrdrb
from pywrdrb.pre.flows import _subtract_upstream_catchment_inflows

from config import DATASET_NAMES

REDO_INFLOW_CALCULATION = False
REDO_INFLOW_PREDICTION = False
REDO_DIVERSION_EXTRAPOLATION = True
REDO_DIVERSION_PREDICTION = True

# Setup pathnavigator
pn_config = pywrdrb.get_pn_config()
for dataset in DATASET_NAMES:
    f = f"pywrdrb/inputs/{dataset}"
    pn_config[f"flows/{dataset}"] = os.path.abspath(f)
pywrdrb.load_pn_config(pn_config)


if __name__ == "__main__":

    for dataset in DATASET_NAMES[:3]:

        ## Calculate catchment inflows
        if REDO_INFLOW_CALCULATION:
            f = f"pywrdrb/inputs/{dataset}/gage_flow_mgd.csv"
            flow_df = pd.read_csv(
                f, index_col=0,
                parse_dates=True,
            )

            # Iteratively subtract upstream catchment flows
            inflow_df = _subtract_upstream_catchment_inflows(flow_df)
            inflow_df.index.name = "datetime"
            
            # Save the inflow_df to a CSV file
            f = f"pywrdrb/inputs/{dataset}/catchment_inflow_mgd.csv"
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