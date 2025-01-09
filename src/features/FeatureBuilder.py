#### construct windows as features ####
import pandas as pd
import os
import numpy as np
import json
import sys
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from features.aggregate_gait_parameters import aggregate_parameters_from_df


class FeatureBuilder:
    def __init__(
        self,
        data_base_path,
        sub_list,
        run_list,
        run_name,
        drop_turning_interval,
        add_static_features=False,
    ):
        """Build and save window-based gait parameter features

        Args:
            data_base_path (str): path to data
            sub_list (list): list of subject IDs
            runs (list): list of runs
            drop_turning_interval (Boolean): whether to drop turning interval strides
        """
        self.data_base_path = data_base_path
        self.save_feature_path = os.path.join(
            data_base_path, "processed", "features_no_abs_SI"
        )
        if not os.path.exists(self.save_feature_path):
            os.makedirs(self.save_feature_path)

        self.sub_list = sub_list
        self.run_list = run_list
        self.run_name = run_name

        self.drop_turning_interval = drop_turning_interval
        self.add_static_features = add_static_features

    def get_valid_strides_df(self, data_base_path, run, sub, foot):
        # build the path
        data_path = os.path.join(
            data_base_path, "processed", sub, run, f"{foot}_foot_core_params.csv"
        )

        # read the data
        df = pd.read_csv(data_path)

        # insert subject info
        df["sub"] = sub
        df["foot"] = foot

        if self.add_static_features:
            # read sub_info
            sub_info = pd.read_csv(
                os.path.join(data_base_path, "raw", "subject_info.csv")
            )
            df["sub_height"] = sub_info[sub_info["sub"] == sub]["height(cm)"].item()
            df["sub_leg_length"] = sub_info[sub_info["sub"] == sub][
                "leg_length(cm)"
            ].item()

        df[self.run_name] = run

        # drop outliers if present
        df = df[
            df["is_outlier"] == False
        ].copy()  # this also removes na value rows because they have been marked as outliers before
        if self.drop_turning_interval:
            df = df[df["turning_interval"] == False].copy()
        return df

    def merge_left_right(self, df_left, df_right):
        """Merge dataframes from left and right foot, sort by timestamps

        Args:
            df_left (dataframe): df for left foot
            df_right (dataframe): df for right foot

        Returns:
            pandas DataFrame: merged dataframe
        """
        merged = pd.concat([df_left, df_right], axis=0).sort_values(by="timestamp")
        return merged

    def construct_windows(self, df, window_sz=10, window_slide=2):
        """Generate a list of dataframe windows

        Args:
            df (dataframe): input dataframe with all data
            window_sz (int, optional): window size. Defaults to 10.
            window_slide (int, optional): step of sliding windows. Defaults to 2.

        Returns:
            list: a list of dataframe windows
        """

        windowed_dfs = []
        start_idx = 0
        end_idx = window_sz
        while end_idx <= len(df) - 1:
            windowed_dfs.append(df[start_idx:end_idx].copy())
            start_idx += window_slide
            end_idx += window_slide

        # add window number (for exporting windowed strid-by-stride data)
        for i in range(len(windowed_dfs)):
            windowed_dfs[i]["window_num"] = i

        return windowed_dfs

    def collect_sub_run_dfs(self):  # collect subjects and runs in dfs
        """Collect all individual dataframes of subs and runs in a list"""
        self.sub_run_df_list = []  # collect subjects and runs in individual dfs
        for sub in self.sub_list:
            for run in self.run_list:
                df_left_right_list = []
                for foot in ["left", "right"]:
                    df_foot = self.get_valid_strides_df(
                        self.data_base_path,
                        run=run,
                        sub=sub,
                        foot=foot,
                    )
                    # self.sub_run_df_list.append(df_foot)
                    df_left_right_list.append(df_foot)
                merged = self.merge_left_right(
                    df_left_right_list[0], df_left_right_list[1]
                )
                merged[self.run_name] = run
                self.sub_run_df_list.append(merged)

    def collect_all_strides(self):
        """Collect and save all strides from subjects and runs"""

        self.collect_sub_run_dfs()

        df_all = pd.concat(self.sub_run_df_list)
        if self.add_static_features:
            df_static = pd.read_csv(
                os.path.join(data_base_path, "raw", "subject_info.csv")
            )
            df_all = pd.merge(df_all, df_static, on="sub")

        df_all.to_csv(os.path.join(self.save_feature_path, "df_all.csv"), index=False)

    def collect_all_windows(self, window_sz, window_slide):
        # construct windows within each subject and run
        self.collect_sub_run_dfs()

        self.windows_df_all_list = []  # collect all windows of strides
        # windows_df_all_list = []
        for df in self.sub_run_df_list:
            windows = self.construct_windows(
                df, window_sz=window_sz, window_slide=window_slide
            )
            self.windows_df_all_list.extend(windows)

        # save the windowed strides
        windows_df_all = pd.concat(self.windows_df_all_list)
        windows_df_all.to_csv(
            os.path.join(self.save_feature_path, "windows_df_all.csv"), index=False
        )

    def build_features(
        self,
        data_base_path,
        window_sz,
        window_slide,
        aggregate_windows=True,
        save_unwindowed_df=True,
    ):
        """Build features based on windows and (optionally) static features

        Args:
            data_base_path (str): data base path
            window_sz (int): size of windows
            window_slide (int): step size of sliding windows
            aggregate_windows (bool, optional): whether to aggregate windows. Defaults to True.
            add_static_features (bool, optional): whether to add static features. Defaults to False.
            save_unwindowed_df (bool, optional): whether to save unwindowed dataframe. Defaults to True.
        """

        if save_unwindowed_df:
            self.collect_all_strides()

        # aggregate parameters and save it to csv for other methods such as SVM
        if aggregate_windows:
            self.collect_all_windows(window_sz, window_slide)
            agg_dat = [
                aggregate_parameters_from_df(df, abs_SI=False)
                for df in self.windows_df_all_list
            ]
            all_windows_df = pd.concat(agg_dat)
            all_windows_df.reset_index(drop=True, inplace=True)
            all_windows_df.dropna(inplace=True)  # in case SI parameters are NaN

            if self.add_static_features:
                sub_info_df = pd.read_csv(
                    os.path.join(data_base_path, "raw", "subject_info.csv")
                )
                sub_info_df = sub_info_df.filter(
                    items=[
                        "sub",
                        "sex",
                        "age",
                        "height(cm)",
                        "leg_length(cm)",
                        "weight(kg)",
                    ]
                )
                all_windows_df = pd.merge(all_windows_df, sub_info_df, on="sub")

            path = os.path.join(
                self.save_feature_path, f"agg_windows_{window_sz}_{window_slide}.csv"
            )
            all_windows_df.to_csv(path, index=False)
            print(f"Saved aggregated windowed features to {path}.")
            # sys.exit()

    def collect_across_sessions(self):
        """Collect aggregated gait parameters across sessions for all subs and runs and save as .csv"""

        across_sessions_dfs = []  # aggregated from both feet
        across_sessions_LR_dfs = {}  # aggregated from left and right feet separately
        across_sessions_LR_dfs["left"] = []
        across_sessions_LR_dfs["right"] = []
        for sub in self.sub_list:
            for run in self.run_list:
                # collect aggregated gait parameters
                agg_df = pd.read_csv(
                    os.path.join(
                        self.data_base_path,
                        "processed",
                        sub,
                        run,
                        "aggregate_params.csv",
                    )
                )
                agg_df["sub"] = sub
                agg_df[self.run_name] = run
                across_sessions_dfs.append(agg_df)

                # collect aggregated gait parameters for left and right feet separately
                for foot in ["left", "right"]:
                    agg_foot_df = pd.read_csv(
                        os.path.join(
                            self.data_base_path,
                            "processed",
                            sub,
                            run,
                            f"{foot}_foot_aggregate_params.csv",
                        )
                    )
                    agg_foot_df["sub"] = sub
                    agg_foot_df[self.run_name] = run
                    across_sessions_LR_dfs[foot].append(agg_foot_df)

        # save collected features
        across_sessions_df = pd.concat(across_sessions_dfs)
        across_sessions_df.to_csv(
            os.path.join(self.save_feature_path, "across_sessions_all.csv"), index=False
        )
        across_sessions_LR_df = {}
        for foot in ["left", "right"]:
            across_sessions_LR_df[foot] = pd.concat(across_sessions_LR_dfs[foot])
            across_sessions_LR_df[foot].to_csv(
                os.path.join(self.save_feature_path, f"across_sessions_{foot}.csv"),
                index=False,
            )
        print(f"Saved aggregated across sessions data to {self.save_feature_path}.")


if __name__ == "__main__":
    sub_list = [
        "imu0001",
        "imu0002",
        "imu0003",
        "imu0006",
        "imu0007",
        "imu0008",
        "imu0009",
        # "imu0010",  # only has visit 1
        "imu0011",
        "imu0012",
        "imu0013",
        # "imu0014",  # only has visit 1
    ]

    run_list = [
        "visit1",
        "visit2",
    ]

    dataset = "data_charite"

    with open("path.json") as f:
        paths = json.loads(f.read())
    data_base_path = paths[dataset]

    window_sz = 10
    window_slide = 2

    feature_builder = FeatureBuilder(
        data_base_path,
        sub_list,
        run_list,
        run_name="visit",
        drop_turning_interval=False,
        add_static_features=False,
    )
    feature_builder.build_features(
        data_base_path,
        window_sz,
        window_slide,
        aggregate_windows=True,
        save_unwindowed_df=True,
    )
