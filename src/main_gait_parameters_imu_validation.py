import json
import os
#from features.build_features import build_features
#from features.postprocessing import mark_processed_data
from LFRF_parameters import pipeline_playground 
#from features import aggregate_gait_parameters


### PARAMS START ###
sub_list = [
    "physilog",
    # "xsens1",
    # "xsens2"
]
runs = [
    "slow",
    "normal",
    "fast"
]
dataset = 'data_imu_validation'
with open(os.path.join(os.path.dirname(__file__), '..', 'path.json')) as f:
    paths = json.loads(f.read())
data_base_path = paths[dataset]
interim_base_path = os.path.join(data_base_path, "interim")
processed_base_path = os.path.join(data_base_path, "processed")
## PARAMS END ###

### Execute the Gait Analysis Pipeline ###
pipeline_playground.execute(sub_list, runs, dataset, data_base_path)

