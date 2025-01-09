#### re-organize raw files to fit the pipeline ####

import os
import json
from shutil import copyfile, move

runs = [
        # 'treadmill',
        "gait1",
        "gait2",
        "walk_fast",
        "walk_preferred",
        "walk_slow"
    ]

subs = [
    "pp001",
    "pp002",
    "pp003",
    "pp004",
    "pp005",
    "pp006",
    "pp007",
    "pp008",
    "pp009",
    "pp010"
]

# subs = [
    # "pp001",
    # "pp010",
    # "pp011",
    # "pp028",
    # "pp071",
    # "pp077",
    # "pp079",
    # "pp099",
    # "pp101",
    # "pp105",
    # "pp106",
    # "pp109",
    # "pp112",
    # "pp114",
    # "pp122",
    # "pp123",
    # "pp137",
    # "pp139",
    # "pp145",
    # "pp149",
    # "pp158",
    # "pp165",
    # "pp166"
# ]

with open('path.json') as f:
    paths = json.load(f)
base_dir = paths["data_kiel_val"]

for sub in subs:
    for run in runs:    
        for source in [("imu", "imu"), ("optical", "omc")]:
            print(f"sub: {sub}, run: {run}, source: {source[0]}")
            os.makedirs(os.path.join(base_dir, "raw", sub, run, source[0]), exist_ok=True)
            move(
                os.path.join(base_dir, "raw", sub, source[0], f"{source[1]}_{run}.mat"), 
                os.path.join(base_dir, "raw", sub, run, source[0], f"{source[1]}_{run}.mat")
                )
            # os.rename(
            #     os.path.join(base_dir, "raw", sub, source[0], f"{sub}_{source[1]}_treadmill.mat"), 
            #     os.path.join(base_dir, "raw", sub, source[0], f"{source[1]}_treadmill.mat")
            #     )

            # # remove folders that were created on the wrong level by accident
            # os.rmdir(
            #     os.path.join(base_dir, "raw", sub, source[0])
            # )
