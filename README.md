
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

