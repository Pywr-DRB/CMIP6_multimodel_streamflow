
import os
import pandas as pd
import pywrdrb

from config import DATASET_NAMES


# Setup pathnavigator
pn_config = pywrdrb.get_pn_config()
for dataset in DATASET_NAMES:
    f = f"pywrdrb/inputs/{dataset}"
    pn_config[f"flows/{dataset}"] = os.path.abspath(f)
pywrdrb.load_pn_config(pn_config)


if __name__ == "__main__":

    for dataset in DATASET_NAMES[:3]:

        ## Filenames
        model_json_file = f"pywrdrb/json/{dataset}.json"
        model_output_file = f"pywrdrb/outputs/{dataset}.hdf5"
        
        # Get the start and end dates
        f = f"pywrdrb/inputs/{dataset}/gage_flow_mgd.csv"
        flow_df = pd.read_csv(
            f, index_col=0,
            parse_dates=True,
        )
        start_date = flow_df.index.min().strftime("%Y-%m-%d")
        end_date = flow_df.index.max().strftime("%Y-%m-%d")


        print("#" * 50)
        print(f"Running Pywr-DRB simulation for {dataset}...")
        print(f"  Start date: {start_date}"
              f" | End date: {end_date}")
        print("#" * 50)

        ### Make the model
        mb = pywrdrb.ModelBuilder(
            inflow_type=dataset,
            start_date=start_date,
            end_date=end_date,
            options = {
                'nyc_nj_demand_source':'custom'
            }
        )
        
        mb.make_model()
        mb.write_model(model_json_file)
        
        print(f"Saved model JSON to pywrdrb/json/{dataset}.json")
        
        ### Load the model
        model = pywrdrb.Model.load(model_json_file)
        
        ## Setup recorder
        recorder = pywrdrb.OutputRecorder(
            model=model,
            output_filename=model_output_file,
            parameters=[p for p in model.parameters if p.name]
        )
        
        ### Run
        model.run()
        print("#" * 50)
        print(f"DONE with Pywr-DRB simulation for {dataset}...")
        print("#" * 50)
    
    print(f"Done running Pywr-DRB simulations.")