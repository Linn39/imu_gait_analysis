import os
import json
import pandas as pd
from scipy.io import loadmat


sub_list = [
    "pp010",
    "pp011",
    "pp028",
    "pp071",
    "pp079",
    "pp099",
    "pp105", 
    "pp106",
    # "pp114",  # has no timestamp markers 
    "pp137",
    "pp139", 
    "pp158",
    "pp165",
    # "pp166",  # only two timestamp markers
    "pp077",
    "pp101",
    "pp109",
    "pp112", 
    "pp122",
    "pp123",
    "pp145",
    "pp149"
]

with open(os.path.join(os.path.dirname(__file__), "..", "..", "path.json")) as f:
    paths = json.loads(f.read())
data_base_path = paths["data_kiel"]

# get timestamps for start and end of constant treadmill speed
speed_timestamps_list = []
for sub in sub_list:
    print(sub)
    mat_path = os.path.join(data_base_path, "raw", sub, "treadmill", "imu", "imu_treadmill.mat")
    mat_data = loadmat(mat_path, simplify_cells=True)["data"]  # dictionary of data
    speed_timestamps_list.append(mat_data["markers"])

# save timestamps with subject names
speed_timestamps_df = pd.DataFrame(
    data=speed_timestamps_list,
    columns=["start1", "end1", "start2", "end2"]
)
speed_timestamps_df["sub"] = sub_list
speed_timestamps_df.to_csv(
    os.path.join(
        data_base_path,
        "raw",
        "treadmill_timestamps.csv"
    ),
    index=False
    )
