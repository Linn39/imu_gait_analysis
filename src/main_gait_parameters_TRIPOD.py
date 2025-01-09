import json
import os
from LFRF_parameters import pipeline_playground 


### PARAMS START ###
sub_list = [
    "Sub_AL",
    "Sub_BK",
    "Sub_CP",
    "Sub_DF",
    "Sub_EN",
    "Sub_FZ",
    "Sub_GK",
    "Sub_HA",
    "Sub_KP",
    "Sub_LU",
    "Sub_OD",
    "Sub_PB",
    "Sub_RW",
    "Sub_SN",
    "Sub_YU"
    ]
runs = [
    "PWS", 
    "PWS+20", 
    "PWS-20"
]
dataset = 'data_TRIPOD'
with open(os.path.join(os.path.dirname(__file__), '..', 'path.json')) as f:
    paths = json.loads(f.read())
data_base_path = paths[dataset]
## PARAMS END ###

### Execute the Gait Analysis Pipeline ###
pipeline_playground.execute(sub_list, runs, dataset, data_base_path)


