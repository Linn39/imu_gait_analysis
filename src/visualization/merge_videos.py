import os
import subprocess


def merge_videos(ref_video, sensor_video1, sensor_video2, sensor_video3, output_dir, config):

    # merge 4 videos
    subprocess.call(
        [
            "ffmpeg",
            "-i",
            ref_video,
            "-i",
            sensor_video1,
            "-i",
            sensor_video2,
            "-i",
            sensor_video3,
            "-filter_complex",
            "[0]scale=1920x1080,"
            + "tpad=start_duration="
            + str(max(0, config["imu_ic"] - config["reference_video_ic"]))
            + "s,"
            + "pad=iw+10:ih+10:0:0:darkgray[experiment];"
            + "[1]fps=fps=30,"
            + "scale=1920x1080,"
            + "tpad=start_duration="
            + str(max(0, config["reference_video_ic"] - config["imu_ic"]))
            + "s,"
            + "pad=iw+10:ih:0:0:darkgray[sensor1];"
            + "[2]fps=fps=30,"
            + "scale=1920x1080,"
            + "tpad=start_duration="
            + str(max(0, config["reference_video_ic"] - config["imu_ic"]))
            + "s,"
            + "pad=iw:ih+10:0:0:darkgray[sensor2];"
            + "[3]fps=fps=30,"
            + "scale=1920x1080,"
            + "tpad=start_duration="
            + str(max(0, config["reference_video_ic"] - config["imu_ic"]))
            + "s[sensor3];"
            + "[experiment][sensor1][sensor2][sensor3]xstack=inputs=4:layout=0_0|0_h0|w0_0|w0_h2",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            os.path.join(output_dir, f"merged_sensor_video_{config['sub']}_{config['run']}_long.mp4"),
        ]
    )

    # remove video before IMU initial contact
    subprocess.call(
        [
            "ffmpeg",
            "-i",
            os.path.join(output_dir, f"merged_sensor_video_{config['sub']}_{config['run']}_long.mp4"),
            "-ss",
            str(max(0, config["imu_ic"] - config["reference_video_ic"])),
            "-c",
            "copy",
            "-map",
            "0",
            os.path.join(output_dir, f"merged_sensor_video_{config['sub']}_{config['run']}.mp4"),
        ]
    )

def cut_ref_video(ref_video, sensor_video1, reference_video_ic):
    """cut the reference video to the length of the sensor video
    """
    pass
