import json
import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

module_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."
)  # src folder
if module_path not in sys.path:
    sys.path.append(module_path)
from data_reader.imu import IMU

"""
Functions to select a subset of the strides recorded in each 6-minute session.
"""


def mark_turning_interval(df, interval_size):
    """
    Parameters
    ----------
    df : pandas DataFrame containing core gait parameters from each foot
    interval_size : num. of strides to be identified as 'turning interval' surrounding the explicit turning steps

    Returns
    -------
    pandas DataFrame with added 'turning_interval' column to mark a larger selection of turning steps
    """
    ts = df.index[df["turning_step"] == True].tolist()
    turning_idx = []
    for x in ts:
        turning_idx.extend(np.arange(x - interval_size, x + interval_size + 1))

    turning_idx = np.array(sorted(set(turning_idx)))
    # select only indices within all strides
    all_idx = np.array(range(df.shape[0]))
    turning_idx = turning_idx[np.isin(turning_idx, all_idx)]

    df["turning_interval"] = False
    df.loc[turning_idx, "turning_interval"] = True

    # remove first and last strides in the session as well
    head_tail_slice = list(range(interval_size)) + list(range(-1 * interval_size, 0))
    df.iloc[head_tail_slice, df.columns.get_loc("turning_interval")] = True

    return df


def mark_interrupted_strides(df, sub, run, interrupt_path):
    """
    Mark strides where the participant was interrupted while walking

    Parameters
    ----------
    df : pandas DataFrame containing core gait parameters
    sub: subject number
    run: run (recording session) number
    interrupt_path : path to interruption time range csv file

    Returns
    -------

    """
    interrupt_df = pd.read_csv(interrupt_path)
    df["interrupted"] = False

    for index, row in interrupt_df.iterrows():
        if np.logical_and(
            interrupt_df.loc[index, "sub"] == sub,
            interrupt_df.loc[index, "run"]
            == run[3:],  # remove prefix 'OG_' from column name
        ):
            # mark interrupted strides within the manually documented time interval
            true_idx = df[
                np.logical_and(
                    df["timestamps"] >= interrupt_df.loc[index, "start(s)"],
                    df["timestamps"] <= interrupt_df.loc[index, "end(s)"],
                )
            ].index
            df.loc[true_idx, "interrupted"] = True

    return df


def select_stride_range(df, position, interval):
    """
    Parameters
    ----------
    df :  pandas DataFrame containing core gait parameters from each foot
    position : string marking where to select the strides, 'start', 'end', 'middle'
    interval : interval of strides to be selected. For 'start' & 'end': interger, number of strides;
    for 'middle', a tuple of range of stride index, e.g. (5, 15)

    Returns
    -------
    pandas DataFrame with core gait parameters from selected strides
    """
    # check if truning strides are marked in the input dataframe
    if not "turning_interval" in df.columns:
        print("turning_interval not marked in the input DataFrame. Abort.")
        pass

    # drop outliers and turning intervals
    df_subset = df.drop(
        df.index[np.logical_or(df["is_outlier"] == 1, df["turning_interval"] == 1)]
    )

    # cut start or end
    if position == "start":
        df_subset = df_subset[:interval]
    if position == "end":
        df_subset = df_subset[-interval:]

    # cut in the middle
    if position == "middle":
        df_subset = df_subset[interval[0] : interval[1]]

    return df_subset


def cut_by_stride(params_LF, params_RF, imu_path, save_path):
    """
    Cut IMU signals based on gait events from left- and right foot

    Parameters
    ----------
    params_LF : dataframe of stride-by-stride gait parameters
    params_RF : dataframe of stride-by-stride gait parameters
    imu_path : IMU data path
    save_path : export path

    Returns
    -------

    """

    # get corresponding raw signal df
    imu = IMU(imu_path)

    imu.data["stride_idx_LF"] = None
    imu.data["ic1_LF"] = None
    imu.data["ic2_LF"] = None
    imu.data["is_outlier_LF"] = None
    imu.data["turning_step_LF"] = None
    imu.data["stride_idx_RF"] = None
    imu.data["ic1_RF"] = None
    imu.data["ic2_RF"] = None
    imu.data["is_outlier_RF"] = None
    imu.data["turning_step_RF"] = None

    # LF
    for idx, ic1, ic2, outlier, turning_step in zip(
        params_LF.stride_index,
        params_LF.timestamps,
        params_LF.ic_times,
        params_LF.is_outlier,
        params_LF.turning_step,
    ):
        # select the matching samples for that stride
        indizes = imu.data[
            (imu.data["timestamp"] >= ic1) & (imu.data["timestamp"] <= ic2)
        ].index
        imu.data.loc[indizes, "stride_idx_LF"] = idx
        imu.data.loc[indizes, "ic1_LF"] = ic1
        imu.data.loc[indizes, "ic2_LF"] = ic2
        imu.data.loc[indizes, "is_outlier_LF"] = outlier
        imu.data.loc[indizes, "turning_step_LF"] = turning_step

    # RF
    for idx, ic1, ic2, outlier, turning_step in zip(
        params_RF.stride_index,
        params_RF.timestamps,
        params_RF.ic_times,
        params_RF.is_outlier,
        params_RF.turning_step,
    ):
        # select the matching samples for that stride
        indizes = imu.data[
            (imu.data["timestamp"] >= ic1) & (imu.data["timestamp"] <= ic2)
        ].index
        imu.data.loc[indizes, "stride_idx_RF"] = idx
        imu.data.loc[indizes, "ic1_RF"] = ic1
        imu.data.loc[indizes, "ic2_RF"] = ic2
        imu.data.loc[indizes, "is_outlier_RF"] = outlier
        imu.data.loc[indizes, "turning_step_RF"] = turning_step

    imu.data.to_csv(save_path, index=False)

    stop = 1


def mark_processed_data(runs, sub_list, processed_base_path, interim_base_path):
    for run in runs:
        for sub in sub_list:
            for foot in ["left", "right"]:  # comment out for cut_by_stride
                folder_path = os.path.join(processed_base_path, sub, run)
                df_path = os.path.join(folder_path, foot + "_foot_core_params.csv")
                params_df = pd.read_csv(df_path)
                ### mark turning intervals
                params_df = mark_turning_interval(
                    params_df, 2
                )  # add 'turning interval' column

                ## mark strides that are interrupted during the recording
                params_df = mark_interrupted_strides(
                    params_df,
                    sub,
                    run,
                    os.path.join(interim_base_path, "interruptions.csv"),
                )

                # save updated df
                print(f"Marked processed data from {df_path}.")
                params_df.to_csv(df_path, index=False)


if __name__ == "__main__":
    # params
    # dataset = 'fatigue_dual_task'
    # with open('../../path.json') as f:
    with open(os.path.dirname(__file__) + "/../../path.json") as f:
        paths = json.load(f)
    processed_base_path = paths["processed_pub"]
    interim_base_path = paths["interim_pub"]

    runs = [
        # 'OG_st_control',
        # 'OG_st_fatigue',
        # 'OG_dt_control',
        "OG_dt_fatigue",
    ]

    sub_list = [
        "sub_01",
        "sub_02",
        "sub_03",
        "sub_05",
        "sub_06",
        "sub_07",
        "sub_08",
        "sub_09",
        "sub_10",
        "sub_11",
        "sub_12",
        "sub_13",
        "sub_14",
        "sub_15",
        "sub_17",
        "sub_18",
    ]

    mark_processed_data(runs, sub_list, processed_base_path, interim_base_path)

    # select sensors for cutting raw data
    # sensors = [
    #     "RF"  # "SA"
    # ]
    # for run in runs:
    #     for sub in sub_list:
    #         ### select stride range ###
    #         last_stride_idx = params_df[-1:].index.values[0]
    #         params_df = select_stride_range(params_df, 'middle', (50, last_stride_idx))
    #         params_df = select_stride_range(params_df, 'middle', (50, 100))
    #         add stride_group column
    #         params_df_1["stride_group"] = "50_to_100"
    #         params_df["stride_group"] = "last_50"
    #         params_df = pd.concat([params_df_1, params_df_2])
    #         prefix = '50_up_to_100_'
    #         params_df.to_csv(os.path.join(folder_path, 'stable_gait_speeds', prefix + foot + '_foot_core_params.csv'), index=False)
    #         plt.scatter(params_df['stride_index'], params_df['stride_lengths'])
    #         plt.show()
    #
    #         # ### cut_by_stride start ###
    #         # execute this to label the interim data with the stride numbers from the processed data
    #         params_LF = pd.read_csv(os.path.join(folder_path, 'left' + '_foot_core_params.csv'))
    #         params_RF = pd.read_csv(os.path.join(folder_path, 'right' + '_foot_core_params.csv'))
    #
    #         for sensor in sensors:
    #             interim_path = os.path.join(
    #                 interim_base_path,
    #                 dataset,
    #                 run,
    #                 sub,
    #                 sensor + ".csv"
    #             )
    #             save_folder = os.path.join(
    #                 interim_base_path,
    #                 dataset,
    #                 run,
    #                 sub,
    #                 "cut_by_stride"
    #             )
    #             if not os.path.exists(save_folder):
    #                 os.makedirs(save_folder)
    #             cut_by_stride(params_LF, params_RF, interim_path, os.path.join(save_folder, sensor + ".csv"))
    #         # ### cut_by_stride end ###
