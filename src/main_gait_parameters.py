import json
import os
#from features.build_features import build_features
#from features.postprocessing import mark_processed_data
from LFRF_parameters import pipeline_playground 
#from features import aggregate_gait_parameters


### PARAMS START ###
sub_list = [
   "Sub_FZ"
]
runs = [
"PWS", "PWS+20", "PWS-20"
]
dataset = 'TRIPOD_excerpt'
with open(os.path.join(os.path.dirname(__file__), '..', 'path.json')) as f:
    paths = json.loads(f.read())
data_base_path = paths["data_imu_validation"]
interim_base_path = os.path.join(data_base_path, "interim")
processed_base_path = os.path.join(data_base_path, "processed")
## PARAMS END ###

### Execute the Gait Analysis Pipeline ###
pipeline_playground.execute(sub_list, runs, dataset, data_base_path)

### Aggregate Gait Parameters over all recording session ###
# aggregate_gait_parameters.main(runs, sub_list, processed_base_path)

# ### Mark outliers strides (turning intervals, interrupted strides) ###
# mark_processed_data(runs, sub_list, processed_base_path, interim_base_path)

# ### Build windows ###
# base_path = paths["data_pub"]
# window_sz = 10
# window_slide = 2
# build_features(sub_list, base_path, test, conditions, window_sz, window_slide, 
#                 aggregate_windows=True, add_static_features=True, save_unwindowed_df=True)
