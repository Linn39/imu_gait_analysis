#### anonymize videos ####
import os
import json
import numpy as np
import subprocess


def remove_videos(video_dir):
    """Remove all video files from the data folder of participants

    Args:
        video_dir (str): path leading to the video files
    """
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith(".mp4"):
                os.remove(os.path.join(root, file))


def remove_audio(input_path, output_path):
    """Remove audio from video file

    Args:
        input_path (str): path leading to the input video files
        output_path (str): path leading to the output video files
    """
    command = f"ffmpeg -i {input_path} -c copy -an {output_path}"
    subprocess.call(command, shell=True)


def crop_video(video_path):
    """Crop video files to only show the feet

    Args:
        video_dir (str): path leading to the video files
    """
    # get codec of video
    command_codec = f"ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 {video_path}"
    codec = subprocess.check_output(command_codec, shell=True).decode("utf-8").strip()

    if codec == "h264":
        # crop video
        command_crop = f'ffmpeg -i {video_path} -filter:v "crop=in_w:in_h*7/8:0:in_h/8" -c:v libx265 -crf 17 {video_path.replace(".mp4", "_cropped.mp4")}'
        # command_crop = f'ffmpeg -i {video_path} -filter:v "crop=in_w:in_h*7/8:0:in_h/8" {video_path.replace(".mp4", "_cropped.mp4")}'
        subprocess.call(command_crop, shell=True)
    else:
        print(
            f"Video {video_path} has codec {codec}, update codec in command_crop before cropping."
        )


if __name__ == "__main__":
    with open("path.json") as f:
        paths = json.load(f)

    # # remove videos from subject folders
    # sub_list = [
    #     "imu0001",
    #     "imu0002",
    #     "imu0003",
    #     "imu0006",
    #     "imu0007",
    #     "imu0008",
    #     "imu0009",
    #     "imu0010",  # only has visit 1
    #     "imu0011",
    #     "imu0012",
    #     "imu0013",
    #     "imu0014",  # only has visit 1
    # ]
    # for sub in sub_list:
    #     video_dir = os.path.join(paths["data_charite"], "raw", sub)
    #     remove_videos(video_dir)

    # # remove audio from videos
    # all_video_dir = os.path.join(paths["data_charite"], "raw", "all_videos")
    # for root, dirs, files in os.walk(all_video_dir):
    #     for file in files:
    #         if file.endswith(".mp4") and np.logical_or("v1" in file, "v2" in file):
    #             input_path = os.path.join(root, file)
    #             output_path = os.path.join(root, file.replace("_v", "_visit_"))
    #             remove_audio(input_path, output_path)

    # crop videos to only show the feet
    video_path = os.path.join(
        paths["data_charite"],
        "raw",
        "all_videos",
        "imu0013_visit_1.mp4",
    )
    crop_video(video_path)
