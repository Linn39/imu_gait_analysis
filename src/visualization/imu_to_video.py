import os
import json
import numpy as np
import shutil
import subprocess
import multiprocessing
import matplotlib.pyplot as plt

from data_reader.imu import IMU


def plot(imu, imu_loc):
    """plot IMU accelearation and gyro signals

    Args:
        imu (IMU object): the IMU object
        imu_loc (str): IMU placement (e.g., LF for left foot)

    Returns:
        _type_: _description_
    """
    figure, axarr = plt.subplots(2, sharex=True, figsize=(910 / 96, 616 / 96), dpi=96)
    figure.suptitle(imu_loc)
    # if imu_id:
    #     figure.suptitle("IMU left foot")
    # else:
    #     figure.suptitle("IMU right foot")

    axarr[0].plot(imu.time(), imu.data["AccX"], label="X")
    axarr[0].plot(imu.time(), imu.data["AccY"], label="Y")
    axarr[0].plot(imu.time(), imu.data["AccZ"], label="Z")
    axarr[0].legend(loc="upper left")
    axarr[0].set_ylabel("Acceleration\n[m/s²]")
    axarr[0].grid()

    axarr[1].plot(imu.time(), imu.data["GyrX"], label="X")
    axarr[1].plot(imu.time(), imu.data["GyrY"], label="Y")
    axarr[1].plot(imu.time(), imu.data["GyrZ"], label="Z")
    axarr[1].legend(loc="upper left")
    axarr[1].set_ylabel("Angular Velocity\n[rad/s]")
    axarr[1].grid()

    plt.xlabel("Time [s]")

    return axarr


def generate_images_helper(args):
    generate_images(*args)


def generate_images(imu_id, imu, imu_loc, imu_video_length, tmp_folder):
    """Generate frames for the IMU video

    Args:
        imu_id (int): index of the imu object
        imu (IMU object): the imu object to genrate the frames
        imu_loc (str): IMU placement (e.g., LF for left foot)
        imu_video_length (int): length of the IMU video in seconds
        tmp_folder (str): path to save the generated frames
    """
    imu.acc_to_meter_per_square_sec()
    imu.gyro_to_rad()

    plt.style.use("seaborn-whitegrid")
    axarr = plot(imu, imu_loc)

    n_samples = imu_video_length * 120   # number of samples at 120 Hz
    for i, time in enumerate(imu.time()[np.arange(0, n_samples, 4, dtype=int)]):   
        # first timestamps for testing, with a downsampling step of 4 so we are at 30 Hz
        plt.xlim([time - 3, time + 3])
        vline_0 = axarr[0].axvline(time, color="r")
        vline_1 = axarr[1].axvline(time, color="r")
        plt.savefig(os.path.join(
            tmp_folder,
            str(imu_id) + "_" + str(i).zfill(6) + ".png",
        ))
        vline_0.remove()
        vline_1.remove()


def generate_videos(imu_id, tmp_folder, output_folder):
    """Generate videos using ffmpeg

    Args:
        imu_id (int): index of the imu object
        tmp_folder (str): path where the frames are saved
        output_folder (str): path where the IMU videos is exported
    """
    ffmpeg_calls = [
        [
            "ffmpeg",
            "-framerate",
            "30",  # framerate at 30 Hz, because we already downsampled when plotting the frames
            "-i",
            os.path.join(tmp_folder,f"{imu_id}_%06d.png"),
            "-c:v",
            "copy",
            os.path.join(output_folder, f"imu_{imu_id}.mp4"),
        ]
    ]

    processes = []
    for call in ffmpeg_calls:
        p = subprocess.Popen(call, stdout=None, stderr=None)
        processes.append(p)

    for p in processes:
        p.wait()


def main():
    sub = 'imu0011'
    run = 'visit1'

    with open("../../path.json") as f:
        paths = json.loads(f.read())
    data_dir = os.path.join(paths["data_charite"], "interim", sub, run, "imu")

    imus = [None, None]
    imus[0] = IMU(os.path.join(data_dir, "RF.csv"))
    imus[1] = IMU(os.path.join(data_dir, "LW.csv"))
    imus[2] = IMU(os.path.join(data_dir, "ST.csv"))

    ic_time = 0

    tmp_folder = "./tmp_video_frames"
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    pool = multiprocessing.Pool(2)
    pool.map(generate_images_helper, [e + (ic_time, tmp_folder) for e in enumerate(imus)])

    generate_videos(tmp_folder)

    shutil.rmtree(tmp_folder)


if __name__ == "__main__":
    main()
