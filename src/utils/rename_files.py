import os
import json
import csv
import shutil
import numpy as np


with open("path.json") as f:
    paths = json.load(f)
data_base_path = paths["data_charite"]
zenodo_base_path = paths["data_charite_zenodo"]
raw_base_path = os.path.join(data_base_path, "raw")


def rename_raw_files(raw_base_path):
    """Rename raw IMU data files and image files

    Args:
        data_base_path (str): string to the data base path

    """

    # go through all .csv files recursively and only keep the first two characters of the file name
    for root, dirs, files in os.walk(raw_base_path):
        for file in files:
            # if file starts with any in the list ["LF", "RF", "LW", "RW", "ST"]
            if np.logical_and(
                file.endswith(".csv"),
                file.startswith(tuple(["LF", "RF", "LW", "RW", "ST"])),
            ):
                os.rename(
                    os.path.join(root, file), os.path.join(root, file[:2] + ".csv")
                )
                # rename ST to SA
                if file[:2] == "ST":
                    os.rename(
                        os.path.join(root, file[:2] + ".csv"),
                        os.path.join(root, "SA.csv"),
                    )

            # rename video files to video.mp4
            if file.endswith(".mp4"):
                os.rename(os.path.join(root, file), os.path.join(root, "video.mp4"))

            # rename image files to feet.jpg
            if file.endswith(".jpg"):
                os.rename(
                    os.path.join(root, file),
                    os.path.join(root, "feet.jpg"),
                )


def remove_date_time(raw_base_path):
    """Remove date time information from the IMU .csv files

    Args:
        data_base_path (str): string to the data base path
    """

    for root, dirs, files in os.walk(raw_base_path):
        for file in files:
            if file.endswith(".csv"):
                with open(os.path.join(root, file), "r") as f:
                    reader = csv.reader(f)
                    rows = list(reader)

                rows[4][1] = "YYYY-MM-DD hh:mm:ss"

                with open(os.path.join(root, file), "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)


def copy_files(source_base_path, target_base_path):
    """Copy files from source to target base path

    Args:
        source_base_path (str): string to the source base path
        target_base_path (str): string to the target base path
    """

    for root, dirs, files in os.walk(source_base_path):
        # for dir in dirs:
        #     os.makedirs(os.path.join(target_base_path, dir), exist_ok=True)
        for file in files:
            if file.startswith(
                "radar_plot_healthy_speed2_across_sessions_"
            ) and file.endswith(
                ".pdf"
            ):  # ("conflicted" not in file)
                # Create the corresponding subfolder in the target base folder if it does not exist
                os.makedirs(
                    os.path.join(
                        target_base_path, os.path.relpath(root, source_base_path)
                    ),
                    exist_ok=True,
                )

                shutil.copyfile(
                    os.path.join(root, file),
                    os.path.join(
                        target_base_path,
                        root.replace(source_base_path, "")[
                            1:
                        ],  # remove first / to get relative path
                        file,
                    ),
                )


def remove_ds_store(source_base_path):
    """Remove .DS_Store files from the source base path

    Args:
        source_base_path (str): string to the source base path
    """

    for root, dirs, files in os.walk(source_base_path):
        for file in files:
            if file == ".DS_Store":
                file_path = os.path.join(root, file)
                print(f"Removing {file_path}")
                os.remove(file_path)  # remove the file


#### main ####
if __name__ == "__main__":
    # rename_raw_files(raw_base_path)
    # remove_date_time(raw_base_path)
    # copy_files(
    #     os.path.join(data_base_path, "processed"),
    #     os.path.join(zenodo_base_path, "processed"),
    # )
    remove_ds_store(zenodo_base_path)
