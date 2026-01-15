#!/bin/bash
#SBATCH --job-name=CMIP
#SBATCH --output=./logs/out.out
#SBATCH --error=./logs/err.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive

# Load modules and environment
module load python/3.11.5
source venv/bin/activate

# number of taks
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# run preprocessing using mpi
# mpirun -n $np python3 01_prep_pywrdrb_inputs.py

# run pywrdrb simulations using mpi
# mpirun -n $np python3 run_pywrdrb_simulations.py

# Run plotting with single process
python3 plot_model_results.py

# python3 plot_dataset_pval_tests.py
# python3 calculate_annual_monthly_stats.py
# python3 get_monthly_shift_scenarios.py
# python3 plot_dataset_comparison.py