
Information about the folder/file naming convention and the contents of the files can be found in the [HydroShare README here](https://hydrosource2.ornl.gov/files/SWA9505V3Flow/README_9505V3Flow.txt).

This dataset can be accessed through one of the following approaches:
- [HydroSource Download](https://hydrosource2.ornl.gov/files/SWA9505V3Flow/)
- [Globus (Recommended) Download](https://doi.org/10.13139/OLCF/2318650)


## Workflow

```
cd CMPI6_multimodel_streamflow/
module load python/3.11.5
python -m virtualenv venv
source venv/bin/activate
pip install git+https://github.com/Pywr-DRB/Pywr-DRB.git
```
This workflow requires pywrdrb>=2.1.0.


`00_preprocessing.py`
    Extracts streamflow at Pywr-DRB nodes from the NetCDF files downloaded from Globus. 
    If you are cloning this repo from GitHub, then this script does _not_ need to be used.  This is only used when new NetCDFs are being processed. 
    After this script is run, `gage_flow_mgd.csv` files will be saved in the `pywrdrb/inputs/<dataset_name>` folder. 

`01_prep_pywrdrb_inputs.py`
    Based on the `gage_flow_mgd.csv` files from each dataset (`pywrdrb/inputs/<dataset_name>`) this script will generate all of the supplemental inputs needed for a Pywr-DRB run, including:
        - catchment inflows
        - predicted inflows
        - extrapolated diversions
        - predicted diversions
    
`02_run_pywrdrb_simulations.py`
    This script will run Pywr-DRB simulations using all of the processed inputs.  


> All of the scripts titled `S*_` are being used by Trevor to determine a subset of climate scenarios and should be considered under development. 


