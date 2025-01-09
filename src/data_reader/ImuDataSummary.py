# summarize IMU data

import os
import json
import pandas as pd
import numpy as np
import glob

from data_reader.SubjectInfo import *
from data_reader.imu import IMU


class ImuDataSummary:
    def __init__(self, dataset, sub_list, run_list, parameter_list):
        self.sub_list = sub_list
        self.run_list = run_list
        self.parameter_list = parameter_list

        # get paths to data
        with open("path.json") as f:
            paths = json.load(f)

        if dataset == "charite":
            data_base_path = paths["data_charite"]

        self.raw_base_path = os.path.join(data_base_path, "raw")
        self.interim_base_path = os.path.join(data_base_path, "interim")
        self.processed_base_path = os.path.join(data_base_path, "processed")

    #### IMU data quality control ####

    # check for data completeness
    def check_imu_files(self, location_list):
        """check if all imu files are present"""
        for sub in self.sub_list:
            for run in self.run_list:
                #### check if processed data files are present
                files_to_check = [
                    "left_foot_core_params",
                    "right_foot_core_params",
                    "left_foot_aggregate_params",
                    "right_foot_aggregate_params",
                    "aggregate_params",
                ]
                missing_files = []
                for file in files_to_check:
                    file_path = os.path.join(
                        self.processed_base_path, sub, run, f"{file}.csv"
                    )
                    if not os.path.exists(file_path):
                        missing_files.append(file_path)

                if missing_files:
                    print(
                        f"Missing processed data: {sub}, {run}: {', '.join(missing_files)}"
                    )

                # raw image files for the feet
                raw_image_path = glob.glob(
                    f"{self.raw_base_path}/{sub}/{run}/image/feet*"
                )
                if not raw_image_path:
                    print(f"Feet image file for {sub}, {run} does not exist!")

                # video files
                raw_video_path = glob.glob(
                    f"{self.raw_base_path}/all_videos/{sub}_*{run[-1]}.mp4"
                )
                if not raw_video_path:
                    print(f"Video file for {sub}, {run} does not exist!")

                for location in location_list:
                    #### check if raw data files are present
                    # imu files
                    raw_data_path = os.path.join(
                        self.raw_base_path, sub, run, "imu", f"{location}.csv"
                    )
                    if not os.path.exists(raw_data_path):
                        print(f"Raw data file {raw_data_path} does not exist!")

                    #### check if interim data files are present
                    # imu files
                    interim_data_path = os.path.join(
                        self.interim_base_path, sub, run, "imu", f"{location}.csv"
                    )
                    if not os.path.exists(interim_data_path):
                        print(f"Interim data file {interim_data_path} does not exist!")

                    # image files
                    interim_image_path = glob.glob(
                        f"{self.interim_base_path}/{sub}/{run}/imu/*{location}.png"
                    )
                    if not interim_image_path:
                        print(
                            f"Interim image file for {sub}, {run}, {location} does not exist!"
                        )

    #### IMU raw data summary ####
    def append_imu_to_df(self, df, sub, run, imu_loc, imu):
        """append imu statistical data to dataframe at the last row

        Args:
            df (DataFrame): the dataframe to be appended
            imu (IMU object): the imu whose information is to be appended
        """
        # get duration of recording
        duration = imu.time()[-1]  # time is zero-based already when reading the data
        # get acceleration magnitude
        imu.acc_to_meter_per_square_sec()
        acc_mag = imu.accel_mag()
        # get gyro magnitude
        gyro_mag = imu.gyro_mag()

        # append data to dataframe
        df.loc[len(df)] = [
            sub,
            imu_loc,
            run,
            duration,
            acc_mag.mean(),
            gyro_mag.mean(),
        ]

    def imu_raw_data_summary(self, imu_locations):
        # load imu data from the 4 walks and the fatigue exercise
        cols = ["sub", "imu_loc", "run", "duration", "acc_mag_mean", "gyro_mag_mean"]
        walk_imu_df = pd.DataFrame(columns=cols)
        exercise_imu_df = pd.DataFrame(columns=cols)
        entire_session_imu_df = pd.DataFrame(columns=cols)

        for sub in self.sub_list:
            for location in imu_locations:
                # load walking data
                for run in self.run_list:
                    imu_walk = IMU(
                        os.path.join(
                            self.interim_base_path, sub, run, "imu", f"{location}.csv"
                        )
                    )
                    self.append_imu_to_df(walk_imu_df, sub, run, location, imu_walk)

        # concat all dataframes and save to csv
        imu_stats_df = pd.concat([walk_imu_df, exercise_imu_df, entire_session_imu_df])
        imu_stats_df.to_csv(
            os.path.join(self.processed_base_path, "imu_stats.csv"), index=False
        )

        # aggregate imu data across participants
        imu_stats_mean_df = (
            imu_stats_df.groupby(["run", "imu_loc"], sort=False)
            .mean()
            .round(2)
            .add_suffix("_mean")
            .reset_index()
        )
        imu_stats_std_df = (
            imu_stats_df.groupby(["run", "imu_loc"], sort=False)
            .std()
            .round(2)
            .add_suffix("_std")
            .reset_index()
        )

        # merge mean and std dataframes
        imu_stats_summary_df = pd.merge(
            imu_stats_mean_df, imu_stats_std_df, on=["run", "imu_loc"], how="inner"
        )
        # sort columns
        imu_stats_summary_df = imu_stats_summary_df[
            [
                "imu_loc",
                "run",
                "duration_mean",
                "duration_std",
                "acc_mag_mean_mean",
                "acc_mag_mean_std",
                "gyro_mag_mean_mean",
                "gyro_mag_mean_std",
            ]
        ]

        # sort rows by imu_loc column
        imu_stats_summary_df = imu_stats_summary_df.sort_values(
            by=["imu_loc"], ascending=[True]
        )

        # save to csv
        imu_stats_summary_df.to_csv(
            os.path.join(self.processed_base_path, "imu_stats_summary.csv"), index=False
        )

        # print to console
        print("IMU raw data summary:")
        print(imu_stats_summary_df.to_string(index=False))

    #### gait parameters summary ####
    def gait_params_summary(self):
        df_list = []
        summary_data = {}
        for run in self.run_list:
            for sub in self.sub_list:
                folder_path = os.path.join(self.processed_base_path, sub, run)
                stride_count = 0
                for foot in ["left", "right"]:
                    core_df = pd.read_csv(
                        os.path.join(folder_path, f"{foot}_foot_core_params.csv")
                    )
                    valid_df = core_df[core_df["is_outlier"] == False]
                    stride_count += valid_df.shape[0]  # count number of valid strides
                agg_df = pd.read_csv(os.path.join(folder_path, "aggregate_params.csv"))
                agg_df["run"] = run
                agg_df["sub"] = sub
                agg_df["stride_count_avg"] = stride_count
                df_list.append(agg_df)

        self.all_agg_df = pd.concat(df_list)[
            self.parameter_list + ["run", "sub", "stride_count_avg"]
        ]
        # save to csv
        self.all_agg_df.to_csv(
            os.path.join(self.processed_base_path, "gait_params_per_person.csv"),
            index=False,
        )

        # calculate mean and std for each parameter
        means = self.all_agg_df.groupby("run", sort=False).mean().round(2)
        std = self.all_agg_df.groupby("run", sort=False).std().round(2)
        # replace column names to indicate std
        std.columns = (
            std.columns.str.replace("avg", "std")
            .str.replace("SI", "SI_std")
            .str.replace("CV", "CV_std")
        )

        # construct summary dataframe
        summary_df = pd.merge(means, std, left_index=True, right_index=True)
        summary_df.sort_index(axis=1, inplace=True)

        # save summary dataframe to csv
        summary_df.to_csv(
            os.path.join(self.processed_base_path, "gait_params_summary.csv")
        )

        print("Gait parameter summary:")
        print(summary_df.transpose().to_string())


#### main ####
if __name__ == "__main__":
    run_list = ["visit1", "visit2"]

    charite_list = [
        "imu0001",
        "imu0002",
        "imu0003",
        "imu0006",
        "imu0007",
        "imu0008",
        "imu0009",
        "imu0010",  # only has visit 1
        "imu0011",
        "imu0012",
        "imu0013",
        "imu0014",  # only has visit 1
    ]

    parameter_list = ["stride_lengths_avg", "speed_avg"]

    data_summary = ImuDataSummary(
        charite_list, sub_list=run_list, parameter_list=parameter_list
    )
    data_summary.imu_raw_data_summary()  # summarize raw IMU data
    data_summary.gait_params_summary()  # summarize gait parameters
