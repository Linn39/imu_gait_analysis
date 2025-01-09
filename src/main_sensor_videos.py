#### make sensor videos ####
import os
import json
import shutil

from data_reader.imu import IMU
from visualization.imu_to_video import *
from visualization.merge_videos import *

config = {}
config["sub"] = 'imu0013'
config["run"] = 'visit2'
config["imu_video_length"] = 15   # length of the imu video in seconds (including the part before initial contact)
config["reference_video_ic"] = 0   # start of stance phase in seconds (always 0, the videos are already trimmed to inital contact)
config["imu_ic"] = 7.5            # start of stance phase in seconds

#### set paths ####
with open("path.json") as f:
    paths = json.loads(f.read())
ref_video_dir = os.path.join(paths["data_charite"], "raw", config["sub"], config["run"], "image")
imu_data_dir = os.path.join(paths["data_charite"], "interim", config["sub"], config["run"], "imu")
output_dir = os.path.join(paths["data_charite"], "processed", "sensor_video")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)  # create path if not exists

#### save config ####
with open(os.path.join(output_dir, f"config_{config['sub']}_{config['run']}.json"), "w") as f:
    json.dump(config, f, indent=4)

# #### generate imu videos ####
imu_loc_ls = ["RF", "RW", "ST"]

tmp_folder = os.path.join(output_dir, "tmp_video_frames")
if not os.path.exists(tmp_folder):
    os.mkdir(tmp_folder)

sensor_videos = []
for imu_idx in range(len(imu_loc_ls)):
    imu = IMU(os.path.join(imu_data_dir, f"{imu_loc_ls[imu_idx]}.csv"))
    # generate_images(imu_idx, imu, imu_loc_ls[imu_idx], config["imu_video_length"], tmp_folder)
    # generate_videos(imu_idx, tmp_folder, output_dir)
    
    sensor_videos.append(
        os.path.join(
            output_dir,
            f"imu_{imu_idx}.mp4",
        ))  # path to sensor videos

shutil.rmtree(tmp_folder)  # remove imu images created for the video

#### merge videos ####
# sensor_video = "RF_imu.mp4"
reference_video = os.path.join(output_dir, f"video_trim_{config['sub']}_{config['run']}.mp4")
output_path = os.path.join(output_dir, f"merged_sensor_video_{config['sub']}_{config['run']}.mp4")

merge_videos(
    reference_video, 
    sensor_videos[0], 
    sensor_videos[1], 
    sensor_videos[2],
    output_dir,
    config,
    )
