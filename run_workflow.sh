#!/bin/bash
#SBATCH --job-name=CMIP
#SBATCH --output=./logs/out.out
#SBATCH --error=./logs/err.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive

module load python/3.11.5
source venv/bin/activate

np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Pywr-DRB workflow. Only needed when new datasets are added; the processed inputs
# are tracked in the repo and the simulation outputs are not used by the S* scripts.
# python P1_extract_netcdf_flows.py
# mpirun -n $np python P2_prep_pywrdrb_inputs.py
# mpirun -n $np python P3_run_pywrdrb_simulations.py
# python P4_plot_nyc_storages.py

# Scenario selection workflow (S2 must precede S3; S3 must precede S4 and S5)
python S1_compare_historic_data.py
python S2_calculate_annual_monthly_stats.py
python S3_find_scenarios.py
python S4_plot_scenarios.py
python S5_plot_annual_flows.py
