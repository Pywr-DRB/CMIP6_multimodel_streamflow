"""
Run a Pywr-DRB simulation for every dataset in pywrdrb/inputs.

Writes the model definition to pywrdrb/json/<dataset>.json and results to
pywrdrb/outputs/<dataset>.hdf5. Datasets are split across MPI ranks:
    mpirun -n <N> python P3_run_pywrdrb_simulations.py
"""
import os
import pandas as pd
import pywrdrb
from mpi4py import MPI
from config import DATASET_NAMES, INPUT_DIR, ROOT_DIR

JSON_DIR = os.path.join(ROOT_DIR, 'pywrdrb', 'json')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'pywrdrb', 'outputs')

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
    local_rank_datasets = [d for i, d in enumerate(DATASET_NAMES) if i % size == rank]
else:
    local_rank_datasets = DATASET_NAMES


if __name__ == "__main__":
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Rank {rank} running {len(local_rank_datasets)} datasets through Pywr-DRB...")

    for dataset in local_rank_datasets:
        model_json_file = os.path.join(JSON_DIR, f"{dataset}.json")
        model_output_file = os.path.join(OUTPUT_DIR, f"{dataset}.hdf5")

        # Simulate the full period covered by the dataset
        flow_df = pd.read_csv(os.path.join(INPUT_DIR, dataset, "gage_flow_mgd.csv"),
                              index_col=0, parse_dates=True)
        start_date = flow_df.index.min().strftime("%Y-%m-%d")
        end_date = flow_df.index.max().strftime("%Y-%m-%d")
        print(f"Running Pywr-DRB for {dataset}: {start_date} to {end_date}")

        mb = pywrdrb.ModelBuilder(
            inflow_type=dataset,
            start_date=start_date,
            end_date=end_date,
            options={'nyc_nj_demand_source': 'custom'},  # historic, custom, or constant
        )
        mb.make_model()
        mb.write_model(model_json_file)

        model = pywrdrb.Model.load(model_json_file)

        # The recorder attaches to the model on creation and writes the HDF5 during run()
        recorder = pywrdrb.OutputRecorder(
            model=model,
            output_filename=model_output_file,
            parameters=[p for p in model.parameters if p.name]
        )
        model.run()

    print(f"Rank {rank} done running Pywr-DRB simulations.")
