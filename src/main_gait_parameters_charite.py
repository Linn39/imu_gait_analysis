import json
import os

# from features.build_features import build_features
from features.postprocessing import mark_processed_data
from LFRF_parameters import pipeline_playground
from features import aggregate_gait_parameters


### PARAMS START ###
sub_list = [
    "imu0001",
    "imu0002",
    "imu0003",
    "imu0006",
    "imu0007",
    "imu0008",
    "imu0009",
    # "imu0010",   # only has visit 1
    "imu0011",
    "imu0012",
    # "imu0013",    # need to adjust thresholds in pipeline_playground.py and gait_parameters.py
    # "imu0014",   # only has visit 1
]
runs = [
    "visit1",
    "visit2",
]
dataset = "data_charite"
with open(os.path.join(os.path.dirname(__file__), "..", "path.json")) as f:
    paths = json.loads(f.read())
data_base_path = paths[dataset]
interim_base_path = os.path.join(data_base_path, "interim")
processed_base_path = os.path.join(data_base_path, "processed")
## PARAMS END ###

### Execute the Gait Analysis Pipeline ###
# pipeline_playground.execute(sub_list, runs, dataset, data_base_path)

### Mark outliers strides (turning intervals, interrupted strides) ###
mark_processed_data(runs, sub_list, processed_base_path, interim_base_path)

### Aggregate Gait Parameters over all recording session ###
aggregate_gait_parameters.main(runs, sub_list, processed_base_path, abs_SI=True)
