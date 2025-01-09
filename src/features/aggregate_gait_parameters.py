import os
import json
import sys

import pandas as pd
import numpy as np
from scipy.stats import variation


def aggregate_parameters_from_df(
    df, select_strides=False, onesided=False, abs_SI=False
):
    """Aggregates stride-by-stride gait parameters from a dataframe

    Args:
        df (DataFrame): dataframe containing stride-by-stride gait parameters
        select_strides (bool, optional): Whether to select first n strides to be analyzed. Defaults to False.
        onesided (bool, optional): Whether data is from only one foot. Defaults to False.
        abs_SI (bool, optional): Whether to take absolute value of symmetry index (SI). Defaults to False.

    Returns:
        DataFrame: dataframe with one row of aggregated gait parameters
    """

    print(".", end="")
    df.reset_index(inplace=True, drop=True)
    # print(
    #     f"sub: {df.loc[df.index[0], 'sub']}, window: {df.loc[df.index[0], 'window_num']}"
    # )
    df.reset_index(inplace=True)
    # load data and filter out all outliers
    df = df[df.is_outlier != 1]  # filter outliers
    if "turning_interval" in df.columns:
        df = df[df.turning_interval != 1]  # filter turning strides
    if "interrupted" in df.columns:
        df = df[df.interrupted != 1]  # filter interrupted strides

    if df.empty:  # if all strides are excluded from the dataframe
        return

    if select_strides:
        # select only first n stride to be aggregated
        df = df.iloc[0:30]

    # get gait parameters from df
    core_params = [
        "stride_length",
        "clearance",
        "stride_time",
        "swing_time",
        "stance_time",
        "stance_ratio",
    ]
    df_param = df.filter(items=core_params)

    # calculate cadence and speed for single foot
    df_param["cadence"] = 120 / df_param["stride_time"]  # cadence in (step / min)
    df_param["speed"] = df_param["stride_length"] / df_param["stride_time"]

    avg_list = df_param.mean().tolist()
    CV_list = variation(df_param, axis=0).tolist()
    aggregate_list = avg_list + CV_list
    aggregate_params = pd.DataFrame(
        columns=[x + "_avg" for x in core_params + ["cadence", "speed"]]
        + [x + "_CV" for x in core_params + ["cadence", "speed"]]
    )
    aggregate_params.loc[0] = aggregate_list

    if onesided:
        all_aggregate_params = aggregate_params
    else:
        # calculate symmetry index (SI) if data is not one-sided
        idx_left = df.index[df["foot"] == "left"]
        idx_right = df.index[df["foot"] == "right"]
        avg_list_left = df_param.loc[idx_left].mean().tolist()
        avg_list_right = df_param.loc[idx_right].mean().tolist()

        SI_list = calculate_SI(avg_list_left, avg_list_right, abs_value=abs_SI)
        SI_df = pd.DataFrame(
            columns=[x + "_SI" for x in core_params + ["cadence", "speed"]]
        )
        SI_df.loc[0] = SI_list

        all_aggregate_params = pd.concat([aggregate_params, SI_df], axis=1)

    # carry over information if they exist in the input dataframe
    if "sub" in df.columns:
        all_aggregate_params["sub"] = df["sub"].values[0]
    if "visit" in df.columns:
        all_aggregate_params["visit"] = df["visit"].values[0]
    if "treadmill_speed" in df.columns:
        all_aggregate_params["treadmill_speed"] = df["treadmill_speed"].values[0]

    return all_aggregate_params


def calculate_SI(avg_list_left, avg_list_right, abs_value=False):
    """Calculate symmetry index (SI) for gait parameters

    Args:
        avg_list_left (list): list of gait parameters from left foot
        avg_list_right (list): list of gait parameters from right foot
        abs_value (bool, optional): Whether to take absolute value of SI. Defaults to False.

    Returns:
        DataFrame: dataframe with one row of aggregated gait parameters
    """

    diff_avg = [(i - j) for i, j in zip(avg_list_left, avg_list_right)]
    sum_avg = [sum(x) for x in zip(avg_list_left, avg_list_right)]
    if abs_value:
        SI_list = [abs(x / (0.5 * y)) for x, y in zip(diff_avg, sum_avg)]
    else:
        SI_list = [x / (0.5 * y) for x, y in zip(diff_avg, sum_avg)]
    return SI_list


def aggregate_parameters(
    save_path, prefix="", abs_SI=False, select_strides=False, save=True
):
    """
    aggregate stride by stride gait parameters
    Parameters
    ----------
    save_path : directory to save aggregated results
    prefix : string in front of common file names, e.g. 'group_0_'
    select_strides: Boolean, whether to select a sub-section of the strides for aggregation
    save : Boolean, save file in .csv

    Returns
    -------
    aggregated gait parameters from left-, right foot and both feet

    """

    # aggregate left- and right foot strides separately
    core_params = {}
    aggregate_params = {}
    for side in ["left", "right"]:
        # load stride-by-stride data into dataframe
        try:
            core_params[side] = pd.read_csv(
                os.path.join(save_path, prefix + side + "_foot_core_params.csv"),
                index_col=False,
            )
            core_params[side]["foot"] = side
        except FileNotFoundError as e:
            print(
                "Foot core data not found. For file with prefix, please create it first using the cutting function."
            )
            print(e)
            sys.exit(0)

        aggregate_params[side] = aggregate_parameters_from_df(
            core_params[side], select_strides=select_strides, onesided=True
        )

    # aggregate left- and right foot strides together, including symmetry indes (SI)
    core_params_LR = pd.concat(core_params.values(), ignore_index=True)
    aggregate_params_LR = aggregate_parameters_from_df(
        core_params_LR, select_strides=select_strides, onesided=False, abs_SI=abs_SI
    )

    if save:
        aggregate_params["left"].to_csv(
            os.path.join(save_path, prefix + "left_foot_aggregate_params.csv"),
            index=False,
        )
        aggregate_params["right"].to_csv(
            os.path.join(save_path, prefix + "right_foot_aggregate_params.csv"),
            index=False,
        )
        aggregate_params_LR.to_csv(
            os.path.join(save_path, prefix + "aggregate_params.csv"), index=False
        )
        # print("saved aggregated gait parameters")

    return aggregate_params, aggregate_params_LR


def main(runs, sub_list, processed_base_path, abs_SI=False):
    for run in runs:
        for sub in sub_list:
            ### aggregate entire recording session
            save_path = os.path.join(processed_base_path, sub, run)
            aggregate_params, aggregate_overall = aggregate_parameters(
                save_path, abs_SI=abs_SI, save=True
            )
            print("aggregate " + run + ", " + sub)

            # ###  aggregate by 2-min group
            # save_path = os.path.join(
            #     processed_base_path,
            #     run,
            #     sub,
            #     'groups'
            # )
            # for group_num in ['0', '1', '2']:
            #     group_name = 'group_' + group_num + '_'
            #     aggregate_params, aggregate_overall = aggregate_parameters(save_path, group_name, save=True)
            #     print('aggregate ' + run + ', ' + sub + ', group ' + group_num)
            #
            # ### aggregate by first n strides
            # save_path = os.path.join(
            #     processed_base_path,
            #     run,
            #     sub
            # )
            # aggregate_params, aggregate_overall = aggregate_parameters(save_path,
            #                                                            prefix='first_30_',
            #                                                            select_strides=True,
            #                                                            save=True)
            # print('aggregate ' + run + ', ' + sub)

            ### aggregate by 50 to end (stable gait speed)
            # save_path = os.path.join(
            #     processed_base_path,
            #     run,
            #     sub,
            #     'stable_gait_speeds'
            # )
            #
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # aggregate_params, aggregate_overall = aggregate_parameters(save_path,
            #                                                            prefix='from_50_',
            #                                                            select_strides=True,
            #                                                            save=True)
            # print('aggregate ' + run + ', ' + sub)

            ### aggregate by 50 to 100 (stable gait speed)
            # save_path = os.path.join(
            #     processed_base_path,
            #     run,
            #     sub,
            #     'stable_gait_speeds'
            # )
            #
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # aggregate_params, aggregate_overall = aggregate_parameters(save_path,
            #                                                            prefix='50_up_to_100_',
            #                                                            select_strides=True,
            #                                                            save=True)
            # print('aggregate ' + run + ', ' + sub)

            ### aggregate by last 50 (stable gait speed)
            # save_path = os.path.join(
            #     processed_base_path,
            #     run,
            #     sub,
            #     'stable_gait_speeds'
            # )
            #
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # aggregate_params, aggregate_overall = aggregate_parameters(save_path,
            #                                                            prefix='last_50_',
            #                                                            select_strides=True,
            #                                                            save=True)
            # print('aggregate ' + run + ', ' + sub)
