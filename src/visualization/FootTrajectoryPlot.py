#### plot aggregated trajectories from all strides ####

#### imports ####
from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_style("whitegrid")

from data_reader.imu import IMU
from LFRF_parameters.trajectory_estimation.filter import gyro_threshold_stance


class FootTrajectoryPlot:
    def __init__(self, dataset, subject, runs, label_paretic_side) -> None:
        self.dataset = dataset
        self.subject = subject
        self.runs = runs
        self.label_paretic_side = label_paretic_side  # whether to use parectic or left/right label for the plots

        with open("path.json") as f:
            paths = json.loads(f.read())
        self.data_base_path = paths[dataset]

        if label_paretic_side:
            # read paretic side from metadata
            paretic_side_df = pd.read_csv(
                os.path.join(self.data_base_path, "raw", "observed_paretic_side.csv")
            )
            self.paretic_side = paretic_side_df[paretic_side_df["sub"] == self.subject][
                "paretic_side"
            ].values[0]

            self.figure_suffix = "paretic_side"  # suffix for the plots

        else:
            self.paretic_side = None

            self.figure_suffix = "LR"  # suffix for the plots

    def plot_aggregated_trajectories(self, beautify):
        fs = None  # sampling rate
        if self.dataset == "data_charite":
            fs = 120
        elif self.dataset == "data_kiel":
            fs = 200

        stride_components = {}
        stride_avg = {}
        stride_speeds = {}
        for run in self.runs:
            interim_data_path = os.path.join(
                self.data_base_path, "interim", self.subject, run
            )
            processed_data_path = os.path.join(
                self.data_base_path, "processed", self.subject, run
            )

            # check if cached 3D trajectories are present
            if os.path.exists(
                os.path.join(interim_data_path, "_trajectory_estimation_left.json")
            ) and os.path.exists(
                os.path.join(interim_data_path, "_trajectory_estimation_right.json")
            ):
                ## read data
                imus = {}
                trajectories = {}
                gait_params = {}
                for foot in [("left", "LF"), ("right", "RF")]:
                    data_label = f"{foot[0]}_{run}"
                    # read trajectory
                    trajectories[foot[0]] = pd.read_json(
                        os.path.join(
                            interim_data_path,
                            "_trajectory_estimation_" + foot[0] + ".json",
                        )
                    )

                    # find stance phases from gyro stance threshold
                    IMU_path = os.path.join(interim_data_path, "imu", foot[1] + ".csv")
                    imu = IMU(IMU_path)  # read interim IMU data
                    imu.acc_to_meter_per_square_sec()
                    imu.gyro_to_rad()
                    imus[foot[0]] = imu

                    imu_gyro_thresholds = pd.read_csv(
                        os.path.join(
                            self.data_base_path,
                            "interim",
                            "stance_magnitude_thresholds_manual.csv",
                        )
                    )
                    stance_thresholds = imu_gyro_thresholds[
                        np.logical_and(
                            imu_gyro_thresholds["subject"] == self.subject,
                            imu_gyro_thresholds["run"] == run,
                        )
                    ]
                    stance = gyro_threshold_stance(
                        imu,
                        float(
                            stance_thresholds[
                                f"stance_magnitude_threshold_{foot[0]}"
                            ].values
                        ),
                        int(
                            stance_thresholds[
                                f"stance_count_threshold_{foot[0]}"
                            ].values
                        ),
                    )
                    swing_begins = np.where(
                        np.logical_and(np.logical_not(stance[1:]), stance[:-1])
                    )[0]

                    # filter out strides that were cut too short or too long
                    swing_begins_diff = swing_begins[1:] - swing_begins[:-1]
                    swing_begins = swing_begins[
                        np.append(
                            np.logical_and(
                                swing_begins_diff > 0.5 * fs, swing_begins_diff < 4 * fs
                            ),
                            True,
                        )
                    ]

                    ## read gait parameters to identify valid strides
                    gait_params_df = pd.read_csv(
                        os.path.join(
                            processed_data_path, f"{foot[0]}_foot_core_params.csv"
                        )
                    )
                    # filter outliers
                    gait_params[foot[0]] = gait_params_df[
                        gait_params_df["is_outlier"] == False
                    ]

                    # cut strides by stance in trajectory
                    stride_components[data_label] = []
                    stride_avg[data_label] = []
                    stride_speeds[data_label] = []
                    len_list = []  # save length of the stride

                    # # # debug: plot 3d trajectory
                    # fig = plt.figure()
                    # ax = fig.add_subplot(111, projection="3d")
                    # ax.set_xlabel("X [m]")
                    # ax.set_ylabel("Y [m]")
                    # ax.set_zlabel("Z [m]")

                    for start, end, fo_time, stride_length in zip(
                        gait_params[foot[0]]["timestamp"],
                        gait_params[foot[0]]["ic_time"],
                        gait_params[foot[0]]["fo_time"],
                        gait_params[foot[0]]["stride_length"],
                    ):
                        for swing_begin1, swing_begin2 in zip(
                            swing_begins[:-1], swing_begins[1:]
                        ):
                            # time sequence for the stride: initial contact 1 -> stance in trajectory 1 -> initial contact 2 -> stance in trajectory 2
                            # select valid strides using initial contacts from the valid gait parameters
                            if np.logical_and(
                                trajectories[foot[0]]["time"][swing_begin1] > start,
                                trajectories[foot[0]]["time"][swing_begin1] < end,
                            ):
                                # # get stride trajectory from stance in trajectory 1 to stance in trajectory 2
                                # fo_time_idx = trajectories[foot[0]].index[np.isclose(trajectories[foot[0]]["time"], fo_time)].item()
                                if beautify:
                                    stride_traj = trajectories[foot[0]].loc[
                                        swing_begin1 + 1 : swing_begin2,
                                        ["position_x", "position_y", "position_z"],
                                    ]
                                else:
                                    stride_traj = trajectories[foot[0]].loc[
                                        swing_begin1:swing_begin2,
                                        ["position_x", "position_y", "position_z"],
                                    ]

                                # # option 0: use PCA to get horizontal and vertical components from xyz
                                # pca = PCA(n_components=2)
                                # components_pca = pca.fit_transform(stride_traj[["position_x", "position_y", "position_z"]])

                                # # option 1: to align the xy dimension: use PCA to combine x and y axis, keep z axis as is
                                # pca = PCA(n_components=1)
                                # horiz_positions = pca.fit_transform(stride_traj[["position_x", "position_y"]])[:,0]
                                # vertical_positions = stride_traj["position_z"]
                                # components = np.asarray(
                                #     [horiz_positions, vertical_positions]
                                # ).T

                                # option 2: to align the xy dimension: project to the line of start and end point
                                # get normalized line vector
                                line_vec = np.asarray(
                                    [
                                        stride_traj["position_x"].iloc[-1]
                                        - stride_traj["position_x"].iloc[0],
                                        stride_traj["position_y"].iloc[-1]
                                        - stride_traj["position_y"].iloc[0],
                                    ]
                                )
                                line_vec_norm = line_vec / np.linalg.norm(line_vec)

                                # project all points on the line with the dot product
                                horiz_positions = np.zeros(len(stride_traj))
                                for i in range(len(stride_traj)):
                                    horiz_positions[i] = np.dot(
                                        np.asarray(  # vector from start point to current point
                                            [
                                                stride_traj["position_x"].iloc[i]
                                                - stride_traj["position_x"].iloc[0],
                                                stride_traj["position_y"].iloc[i]
                                                - stride_traj["position_y"].iloc[0],
                                            ]
                                        ),
                                        line_vec_norm,  # normalized line vector to project to
                                    )

                                # use combined x and y axis, keep z axis as is
                                vertical_positions = stride_traj["position_z"]
                                components = np.asarray(
                                    [horiz_positions, vertical_positions]
                                ).T  # components_proj

                                # # # debug: plot 3d trajectory
                                # fig = plt.figure()
                                # ax = fig.add_subplot(111, projection='3d')
                                # ax.plot(
                                #     stride_traj["position_x"],
                                #     stride_traj["position_y"],
                                #     stride_traj["position_z"],
                                #     # label="original 3D trajectory",
                                # )
                                # ax.plot(
                                #     components[:, 0],
                                #     components[:, 0] * 0,
                                #     components[:, 1],
                                #     label="projected 2D trajectory",
                                # )
                                # ax.plot(components_proj[:, 0], components_proj[:, 0] * 0, components_proj[:, 1], label='dot product')
                                # ax.plot(components_pca[:, 0], components_pca[:, 0] * 0, components_pca[:, 1], label='pca 2 components')
                                # ax.set_xlim3d(-0.2, 2.2)
                                # ax.set_ylim3d(-2.2, 0.2)
                                # ax.set_xlabel("X")
                                # ax.set_ylabel("Y")
                                # ax.set_zlabel("Z")
                                # ax.legend()
                                # plt.show()
                                # # plt.close()

                                # plot trajectory
                                if beautify:  # quick fix to remove outliers after PCA
                                    if np.logical_and(
                                        max(abs(components[:, 0] - components[0, 0]))
                                        < 1.7,  # stride length threshold
                                        max(abs(components[:, 1] - components[0, 1]))
                                        < 0.2,  # clearance threshold
                                    ):
                                        components_abs = [
                                            abs(components[:, 0] - components[0, 0]),
                                            abs(components[:, 1] - components[0, 1]),
                                        ]
                                        stride_components[data_label].append(
                                            np.asarray(components_abs)
                                        )
                                        len_list.append(np.shape(components_abs)[1])
                                        stride_speeds[data_label].append(
                                            stride_length / (end - start)
                                        )  # stride speed
                                else:
                                    components_abs = [
                                        abs(components[:, 0] - components[0, 0]),
                                        abs(components[:, 1] - components[0, 1]),
                                    ]
                                    stride_components[data_label].append(
                                        np.asarray(components_abs)
                                    )
                                    len_list.append(np.shape(components_abs)[1])
                                    stride_speeds[data_label].append(
                                        stride_length / (end - start)
                                    )  # stride speed

                    # ax.set_xlim3d(-0.2, 2.2)
                    # ax.set_ylim3d(-2.2, 0.2)
                    # ax.set_xlabel("X [m]")
                    # ax.set_ylabel("Y [m]")
                    # ax.set_zlabel("Z [m]")
                    # ax.legend()
                    # plt.show()
                    # plt.close()

                    cut_x = []
                    cut_y = []
                    strides_padded_x = []
                    strides_padded_y = []
                    for stride in stride_components[data_label]:
                        # option1: cut all strides to the shortest stride
                        cut_x.append(stride[0, : min(len_list)])
                        cut_y.append(stride[1, : min(len_list)])
                        # # option2: pad all strides to the longest one wit NaN
                        # pad_len = max(len_list) - stride.shape[1]  # the amount to be padded
                        # strides_padded_x.append(np.pad(stride[0].astype(float), (0,pad_len), 'constant', constant_values=np.nan))
                        # strides_padded_y.append(np.pad(stride[1].astype(float), (0,pad_len), 'constant', constant_values=np.nan))

                    # calculate mean
                    avg_x = np.mean(cut_x, axis=0)
                    avg_y = np.mean(cut_y, axis=0)
                    # avg_x = np.nanmean(strides_padded_x, axis=0)
                    # avg_y = np.nanmean(strides_padded_y, axis=0)
                    stride_avg[data_label].append(np.asarray([avg_x, avg_y]))

                    # plt.plot(avg_x, avg_y, color='0', linewidth=4, label='average')
                    # plt.title(f"{self.dataset} {self.subject} {run}, {foot[0]} foot, n = {len(stride_components[foot[0]])}")
                    # plt.xlabel("Distance (m)")
                    # plt.ylabel("Height (m)")
                    # plt.legend()
                    # plt.show()

                # plot both feet in one figure
                fig_feet = plt.figure(figsize=(7, 4))
                stride_count = {}

                if self.label_paretic_side:
                    if self.paretic_side == "Left":
                        foot_labels = [
                            ("left", "orange", "paretic"),
                            ("right", "green", "non-paretic"),
                        ]
                    elif self.paretic_side == "Right":
                        foot_labels = [
                            ("left", "green", "non-paretic"),
                            ("right", "orange", "paretic"),
                        ]
                    title = f"Stride side view {self.dataset} {self.subject} {run} \n Paretic side: {self.paretic_side}"
                else:
                    foot_labels = [
                        ("left", "green", "left"),
                        ("right", "orange", "right"),
                    ]
                    title = f"Stride side view {self.dataset} {self.subject} {run}"

                # plot all strides of one foot and one visit
                for foot in foot_labels:
                    stride_count[foot[0]] = len(stride_components[f"{foot[0]}_{run}"])
                    for stride in stride_components[f"{foot[0]}_{run}"]:
                        plt.plot(stride[0], stride[1], color=foot[1], alpha=0.2)
                    plt.plot(
                        stride_avg[f"{foot[0]}_{run}"][0][0, :],
                        stride_avg[f"{foot[0]}_{run}"][0][1, :],
                        color=foot[1],
                        linewidth=4,
                        label=f"Average {foot[2]} foot",  # , speed = {round(np.mean(stride_speeds[f"{foot[0]}_{run}"]), 2)} m/s',
                    )
                    plt.xlabel("Distance (m)")
                    plt.ylabel("Height (m)")
                    plt.title(title)
                    plt.legend()
                    # plt.show()
                    plt.close()

            else:
                print(f"No trajectories for {self.dataset} {self.subject} {run} found.")

        ## plot both visits in one figure
        # create a new dictionary with the modified keys
        if self.label_paretic_side:
            if self.paretic_side == "Left":
                stride_components_paretic = {
                    key.replace("left", "paretic").replace(
                        "right", "non-paretic"
                    ): value
                    for key, value in stride_components.items()
                }
            elif self.paretic_side == "Right":
                stride_components_paretic = {
                    key.replace("left", "non-paretic").replace(
                        "right", "paretic"
                    ): value
                    for key, value in stride_components.items()
                }
            title = f"Stride side view {self.dataset} {self.subject} \n Paretic side: {self.paretic_side}"
        else:
            stride_components_paretic = stride_components
            title = f"Stride side view {self.dataset} {self.subject}"
        fig_runs = plt.figure(figsize=(7, 4))
        for key, key_paretic, color in zip(
            stride_components.keys(),
            stride_components_paretic.keys(),
            ["blue", "green", "orange", "red"],
        ):
            # rename key_paretic for plotting
            key_paretic_for_plotting = " ".join(key_paretic.split("_")).title()

            # plot single strides
            for stride in stride_components[key]:
                plt.plot(stride[0], stride[1], color=color, alpha=0.1)
            plt.plot(
                stride_avg[key][0][0, :],
                stride_avg[key][0][1, :],
                color=color,
                linewidth=4,
                label=f"Average {key_paretic_for_plotting}, Speed = {round(np.mean(stride_speeds[key]), 2)} m/s",
            )
        plt.xlabel("Distance [m]")
        plt.ylabel("Height [m]")
        # plt.title(title)
        plt.legend()
        plt.savefig(
            os.path.join(
                self.data_base_path,
                "processed",
                "figures_trajectory_sideview",
                f"trajectory_sideview_{self.subject}_dot_product_{self.figure_suffix}.pdf",
            ),
            bbox_inches="tight",
        )
        # plt.show()
        plt.close()


if __name__ == "__main__":
    ## select data
    dataset = "data_kiel"  # "data_charite"  #"data_kiel"
    subject = "pp011"  # "pp137"  #"imu0009"
    runs = [
        "treadmill_speed1",
        "treadmill_speed2",
    ]  # ["treadmill_speed1", "treadmill_speed2"]   #["visit1", "visit2"]   # , "visit2"
    beautify = (
        True  # remove outliers from the plot (not really marking outlier in the data)
    )

    foot_trajectory_plot = FootTrajectoryPlot(dataset, subject, runs)
    foot_trajectory_plot.plot_aggregated_trajectories(beautify=True)
