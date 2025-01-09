import json
import os

from LFRF_parameters import pipeline_playground 
from features.postprocessing import mark_processed_data
from features import aggregate_gait_parameters


### PARAMS START ###
control_list = [
    "pp010",
    "pp011",
    "pp028",
    # "pp071",    # part of the RF data is not usable (IMU dropped from the foot)
    "pp079",
    "pp099",
    "pp105", 
    "pp106",
    # "pp114",    # only one constant treadmill speed
    "pp137",
    "pp139", 
    "pp158",
    "pp165",
    # "pp166"     # only one constant treadmill speed
]

stroke_list = [
    "pp077",
    "pp101",
    "pp109",
    "pp112", 
    "pp122",
    "pp123",
    "pp145",
    "pp149",
]

overground_list = [
        # "pp001",
        "pp002",
        "pp003",
        "pp004",
        "pp005",
        "pp006",
        # "pp007",    # l_psis is missing for the optical data
        "pp008",
        # "pp009",
        # "pp010"
    ]

runs = [
    # "treadmill",      # all treadmill data, including changing speed
    "treadmill_speed1",     # constant speed 1
    "treadmill_speed2",     # constant speed 2
    # "gait1",
    # "gait2",
    # "walk_fast",
    # "walk_preferred",
    # "walk_slow"
]
dataset = "data_kiel" # 'data_kiel', "data_kiel_val"
sub_list = stroke_list
with open(os.path.join(os.path.dirname(__file__), '..', 'path.json')) as f:
    paths = json.loads(f.read())
data_base_path = paths[dataset]
interim_base_path = os.path.join(data_base_path, "interim")
processed_base_path = os.path.join(data_base_path, "processed")

## PARAMS END ###

### Execute the Gait Analysis Pipeline ###
pipeline_playground.execute(sub_list, runs, dataset, data_base_path)   # check which pipeline is being excuted!

### Mark outliers strides (turning intervals, interrupted strides) ###
# mark_processed_data(runs, sub_list, processed_base_path, interim_base_path)

### Aggregate Gait Parameters over all recording session ###
# aggregate_gait_parameters.main(runs, sub_list, processed_base_path)
