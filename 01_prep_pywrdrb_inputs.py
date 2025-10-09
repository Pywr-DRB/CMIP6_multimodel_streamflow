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

# Setup pathnavigator
pn_config = pywrdrb.get_pn_config()
for dataset in DATASET_NAMES:
    f = f"pywrdrb/inputs/{dataset}"
    pn_config[f"flows/{dataset}"] = os.path.abspath(f)
pywrdrb.load_pn_config(pn_config)


if __name__ == "__main__":

    for dataset in DATASET_NAMES:
        
        # check if predicted_inflows_mgd.csv and catchment_inflow_mgd.csv already exist
        # skip if they do
        if os.path.exists(f"pywrdrb/inputs/{dataset}/predicted_inflows_mgd.csv") and os.path.exists(f"pywrdrb/inputs/{dataset}/catchment_inflow_mgd.csv"):
            continue
        
        
        ## Calculate catchment inflows
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
        
        ### Predicted flows
        # Get the start and end dates
        start_date = inflow_df.index.min()
        end_date = inflow_df.index.max()

        inflow_predictor = pywrdrb.pre.PredictedInflowPreprocessor(
            flow_type=dataset,
            start_date=start_date,
            end_date=end_date,
        )
        
        # Predict and save 
        inflow_predictor.load()
        inflow_predictor.process()
        inflow_predictor.save()