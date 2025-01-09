import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.transform import Rotation as rot
from scipy.signal import find_peaks, peak_prominences

# A gait cycle is defined to start with stance phase, followed by swing phase
# Thus a proper recording has an initial contact event as first and last event


class GaitParameters:
    def __init__(self, trajectories, gait_events, show_figs, save_fig_directory, initial_contact=0):
        self.initial_contact = initial_contact
        self.trajectories = trajectories
        self.gait_events = gait_events
        self.show_figs = show_figs
        self.save_fig_directory = save_fig_directory
        self.adjust_data()

    def adjust_data(self):
        stance_begin = self.gait_events["stance_begin"]
        stance_end = self.gait_events["stance_end"]

        for side in ["left", "right"]:
            assert len(self.gait_events[side]["times"][stance_begin]) == len(
                self.gait_events[side]["times"][stance_end]
            )

            # if the recording doesn't start with start event
            while (

                self.gait_events[side]["samples"][stance_begin][0]
                >= self.gait_events[side]["samples"][stance_end][0]
            ):
                
                # drop first end event
                self.gait_events[side]["samples"][stance_end] = self.gait_events[side][
                    "samples"
                ][stance_end][1:]
                self.gait_events[side]["times"][stance_end] = self.gait_events[side][
                    "times"
                ][stance_end][1:]

            # if the recording doesn't end with start event
            if (
                self.gait_events[side]["samples"][stance_begin][-1]
                < self.gait_events[side]["samples"][stance_end][-1]
            ):
                print("if") ###### DEBUG: right foot has no gait events
                print(self.gait_events[side]["samples"][stance_begin])
                print(self.gait_events[side]["samples"][stance_end])
                # drop last end event
                self.gait_events[side]["samples"][stance_end] = self.gait_events[side][
                    "samples"
                ][stance_end][:-1]
                self.gait_events[side]["times"][stance_end] = self.gait_events[side][
                    "times"
                ][stance_end][:-1]

            # now there should be exactly one more start event than end- event
            assert (
                len(self.gait_events[side]["times"][stance_begin])
                == len(self.gait_events[side]["times"][stance_end]) + 1
            )

    def summary_raw(self):
        """
        summary of gait parameters from all strides (including outliers)
        @return:
        @rtype:
        """
        stride_length = self.stride_length()
        stride_time = self.stride_time()
        swing_time = self.swing_time()
        stance_time = self.stance_time()
        angle_change = self.angle_change()
        # clearance_min = self.clearance('min')
        # clearance_max = self.clearance('max')
        clearance_simple = self.clearance_simple()

        summary = {"left": None, "right": None}
        stance_begin = self.gait_events["stance_begin"]
        for side in summary.keys():
            summary[side] = pd.DataFrame(
                data={
                    "timestamp": np.array(self.gait_events[side]["times"][stance_begin][:-1]) - self.initial_contact,
                    "stride_length": stride_length[side],
                    # "clearances_min": clearance_min[side],
                    # "clearances_max": clearance_max[side],
                    "clearance": clearance_simple[side],
                    "angle_change": angle_change[side],
                    "stride_time": stride_time[side],
                    "swing_time": swing_time[side],
                    "stance_time": stance_time[side],
                    "stance_ratio": stance_time[side] / stride_time[side],
                    "speed": stride_length[side]/stride_time[side],
                    "fo_time": self.gait_events[side]["times"]["FO"],
                    "ic_time": self.gait_events[side]["times"]["IC"][1:],  # remove the initial contact in Optogait
                    "fo_sample": self.gait_events[side]["samples"]["FO"],
                    "ic_sample": self.gait_events[side]["samples"]["IC"][1:],  # remove the initial contact in Optogait
                }
            )
            summary[side]['stride_index'] = summary[side].index  # keep track of original index for all detected strides
            # # clean up stride length outliers
            # summary[side] = summary[side][(summary[side]["stride_length"] > 0.25)
            #                               & (summary[side]["stride_length"] < 2.5)
            #                               & (summary[side]["stride_time"] < 2)
            #                               & (summary[side]["angle_change"] < 0.1)]

        return summary

    def summary(self):
        """
        find ourliers, returns cleaned up summary of gait parameters
        @return:
        @rtype:
        """
        summary_raw = self.summary_raw()
        summary = {"left": None, "right": None}
        for side in summary_raw.keys():
            summary_raw[side]['is_outlier'] = False
            summary_raw[side]['turning_step'] = False

            # remove outliers by checking for NaN
            summary_raw[side].loc[summary_raw[side].isna().any(axis=1), 'is_outlier'] = True
            summary_raw[side].isna().sum()

            # remove outliers by threshold
            for parameter, sign, threshold in [
                ('angle_change', 'larger_than', 0.2),
                ('stride_length', 'smaller_than', 0.2),
                ('stride_time', 'larger_than', 2),  # 2 seconds for normal gait, 3.5 seconds for severly impaired gait
                ('stance_ratio', 'smaller_than', 0.5),
            ]:
                # check for angle change
                if sign == 'larger_than':
                    summary_raw[side].loc[summary_raw[side][parameter] > threshold, 'is_outlier'] = True
                    if parameter == 'angle_change':
                        summary_raw[side].loc[summary_raw[side][parameter] > threshold, 'turning_step'] = True
                elif sign == 'smaller_than':
                    summary_raw[side].loc[summary_raw[side][parameter] < threshold, 'is_outlier'] = True

                # plot parameter with outliers
                if self.show_figs != 0:
                    fig = plt.figure()
                    plt.scatter(summary_raw[side].index,
                                summary_raw[side][parameter],
                                label=parameter)
                    plt.scatter(summary_raw[side].index[summary_raw[side]['is_outlier'] == True].tolist(),
                               summary_raw[side][summary_raw[side]['is_outlier'] == 1][[parameter]],
                               marker='o', facecolors='none', edgecolors='r', s=150, label='Outlier')
                    plt.axhline(y=threshold, color='r', linestyle='-')
                    plt.title('Remove outliers: ' + parameter + ' (' + sign + ' ' + str(threshold) + ')'
                              '\nTotoal samples: ' + str(summary_raw[side].shape[0]) +
                              ', Valid samples: ' + str(summary_raw[side].shape[0] - summary_raw[side]['is_outlier'].sum()))
                    plt.legend()

                    plt.savefig(os.path.join(self.save_fig_directory, side + '_' + parameter + '_outliers_threshold.png'),
                                    bbox_inches='tight')
                    if self.show_figs == 1:
                        plt.show()
                    plt.close(fig)

            for parameter in [
                'stride_length',
                # 'clearances_min',
                # 'clearances_max',
                'clearance',
                'stride_time',
                'swing_time',
                'stance_time',
                'stance_ratio',
            ]:
                # calculate z score in Gaussian distribution
                z_threshold = 3
                z_scores = np.abs(stats.zscore(summary_raw[side][parameter]))
                summary_raw[side]['z_score_' + parameter] = z_scores

                # plot parameter with outliers
                if self.show_figs != 0:
                    fig = plt.figure()
                    plt.scatter(summary_raw[side].index,
                                summary_raw[side][parameter],
                                label=parameter)
                    plt.scatter(summary_raw[side].index[summary_raw[side]['z_score_' + parameter] > z_threshold].tolist(),
                                summary_raw[side][summary_raw[side]['z_score_' + parameter] > z_threshold][[parameter]],
                                c='orange', alpha=0.4, s=150, label='Z_score_outlier')
                    plt.scatter(summary_raw[side].index[summary_raw[side]['is_outlier'] == True].tolist(),
                                summary_raw[side][summary_raw[side]['is_outlier'] == 1][[parameter]],
                                marker='o', facecolors='none', edgecolors='r', s=150, label='Threshold_outlier')

                    plt.title('Remove outliers: ' + parameter +
                              '\nTotoal samples: ' + str(summary_raw[side].shape[0]) +
                              ', Valid samples: ' + str(summary_raw[side].shape[0] - summary_raw[side]['is_outlier'].sum()))
                    plt.legend()

                    if self.show_figs == 1:
                        plt.show()
                    elif self.show_figs == 2:
                        plt.savefig(os.path.join(self.save_fig_directory, side + '_' + parameter + '_outliers_z_score.png'),
                                    bbox_inches='tight')
                        plt.close(fig)

                summary_raw[side].loc[summary_raw[side]['z_score_' + parameter] > z_threshold, 'is_outlier'] = True

            summary[side] = summary_raw[side]  # export all strides (including outliers)
            # summary[side] = summary_raw[side][summary_raw[side].is_outlier == False]  # only export valid strides
            summary[side] = summary[side].filter(items=[
                'stride_index',
                'timestamp',
                'stride_length',
                # 'clearances_min',
                # 'clearances_max',
                'clearance',
                'stride_time',
                'swing_time',
                'stance_time',
                'stance_ratio',
                'speed',
                'fo_time',
                'ic_time',
                'fo_sample',
                'ic_sample',
                'is_outlier',
                'turning_step',
            ])

        return summary

    def stride_length(self):
        stride_length = {"left": [], "right": []}
        stance_begin = self.gait_events["stance_begin"]
        for side in stride_length.keys():
            for start, end in zip(
                self.gait_events[side]["samples"][stance_begin][:-1],
                self.gait_events[side]["samples"][stance_begin][1:],
            ):
                step = np.array(
                    [
                        self.trajectories[side]["position_x"][end],
                        self.trajectories[side]["position_y"][end],
                        self.trajectories[side]["position_z"][end],
                    ]
                ) - np.array(
                    [
                        self.trajectories[side]["position_x"][start],
                        self.trajectories[side]["position_y"][start],
                        self.trajectories[side]["position_z"][start],
                    ]
                )

                stride_length[side].append(np.linalg.norm(step[0:2]))
        return stride_length

    def stride_time(self):
        stride_time = {"left": [], "right": []}
        stance_begin = self.gait_events["stance_begin"]
        for side in stride_time.keys():
            stride_time[side] = np.array(
                self.gait_events[side]["times"][stance_begin][1:]
            ) - np.array(self.gait_events[side]["times"][stance_begin][:-1])
        return stride_time

    def clearance_simple(self):
        """_summary_
        Get last peak of maximum clearance
        """

        clearance = {"left": [], "right": []}
        clearance_idx = {"left": [], "right": []}
        clearance_idx_prom = {"left": [], "right": []}  # index for first peak for the prominence plot
        stance_begin = self.gait_events["stance_begin"]
        stance_end = self.gait_events["stance_end"]
        for side in clearance.keys():
            missed_intervals = np.zeros(len(self.trajectories[side]["position_z"]), dtype=bool)
            missed_interval_count = 0
            for swing_start, swing_end in zip(
                    self.gait_events[side]["samples"][stance_end],
                    self.gait_events[side]["samples"][stance_begin][1:],
            ):
                # segment strides
                z = np.array(self.trajectories[side]["position_z"])[swing_start:swing_end]
                # look for maximum foot clearance
                peak_idx, _ = find_peaks(z, prominence=0.001)
                peak_values = z[peak_idx]

                if not (np.isnan(peak_idx).any() or peak_idx.size == 0):
                    peak_idx = peak_idx[peak_values.argmax()]  # identify the largest peak if there are multiple ones
                    peak_value = peak_values[peak_values.argmax()]
                    peak_idx_prom = peak_idx  # index for the prominence plot
                else:
                    peak_value = np.nan
                    peak_idx = np.nan
                    peak_idx_prom = np.nan
                    missed_interval_count += 1
                    missed_intervals[swing_start:swing_end + 1] = 1

                clearance[side].append(peak_value)
                clearance_idx[side].append(peak_idx + swing_start)
                clearance_idx_prom[side].append(peak_idx_prom + swing_start)

            clearance[side] = np.asarray(clearance[side])
            clearance_idx[side] = np.asarray(clearance_idx[side])

            # debugging: check prominence thresholds
            if self.show_figs != 0:
                z_all = np.array(self.trajectories[side]["position_z"])

                clearance_idx_prom[side] = np.asarray(clearance_idx_prom[side])
                clearance_idx_prom_valid = clearance_idx_prom[side][np.logical_not(np.isnan(clearance_idx_prom[side]))].astype(int)
                clearance_prom = peak_prominences(
                    z_all, clearance_idx_prom_valid
                )[0]
                fig = plt.figure()
                plt.scatter(clearance_idx_prom_valid, clearance_prom, c='orange')
                plt.title(f'prominences of {side} simple clearance peaks')
                plt.xlabel('sample number')
                plt.ylabel('prominence')
                if self.show_figs == 1:
                    plt.show()
                    # continue
                elif self.show_figs == 2:
                    plt.savefig(os.path.join(self.save_fig_directory, side + '_clearance_prom_thresholds.png'),
                                bbox_inches='tight')
                    plt.close(fig)

            # plot detected clearances
            if self.show_figs != 0:
                fig, ax = plt.subplots(figsize=(20, 5))
                plt.plot(self.trajectories[side]["position_z"])
                plt.plot(clearance_idx[side], clearance[side],
                         marker='x', linestyle='None', label='clearance_peaks_mean')
                plt.vlines(x=self.gait_events[side]["samples"][stance_begin], ymin=-0.02, ymax=0.1,
                           color='c', label='swing_end')
                plt.vlines(x=self.gait_events[side]["samples"][stance_end], ymin=-0.02, ymax=0.1,
                           color='limegreen', label='swing_begin')
                plt.title(f'Peak detection for {side} simple max clearance' +
                          '\n missed intervals: ' + str(missed_interval_count))
                if sum(missed_intervals) > 0:
                    ax.fill_between(range(len(missed_intervals)),
                                    -0.03, 0.11, where=missed_intervals, alpha=0.3)
                    # transform=ax.get_xaxis_transform())
                plt.legend()
                if self.show_figs == 1:
                    plt.show()
                    # continue
                elif self.show_figs == 2:
                    plt.savefig(os.path.join(self.save_fig_directory, side + '_clearance.png'),
                                bbox_inches='tight')
                    plt.close(fig)

        return clearance

    def clearance(self, min_max):
        """
        Minimum foot clearance (MFC) or average maximum clearance
        @param min_max: 'min' or 'max'
        @type min_max: string
        @return:
        @rtype:
        """
        clearance = {"left": [], "right": []}
        clearance_idx = {"left": [], "right": []}
        clearance_idx_prom = {"left": [], "right": []}  # index for first peak for the prominence plot
        stance_begin = self.gait_events["stance_begin"]
        stance_end = self.gait_events["stance_end"]
        for side in clearance.keys():
            missed_intervals = np.zeros(len(self.trajectories[side]["position_z"]), dtype=bool)
            missed_interval_count = 0
            for swing_start, swing_end in zip(
                    self.gait_events[side]["samples"][stance_end],
                    self.gait_events[side]["samples"][stance_begin][1:],
            ):
                # segment strides
                z = np.array(self.trajectories[side]["position_z"])[swing_start:swing_end]
                # look for minimum foot clearance
                if min_max == 'min':
                    peak_idx, _ = find_peaks(-z, prominence=0.002)
                    # peak_idx = peak_idx[-1]  # identify the last peak if there are multiple ones
                    peak_value = z[peak_idx]

                    # fig = plt.figure()
                    # plt.plot(z)
                    # plt.plot(peak_idx, peak_value,
                    #          marker='x', linestyle='None', label='clearance_peaks')
                    # plt.show()

                    if not (np.isnan(peak_idx).any() or peak_idx.size == 0):
                        peak_idx = peak_idx[-1]  # identify the last peak if there are multiple ones
                        peak_value = peak_value[-1]
                        peak_idx_prom = peak_idx  # index for the prominence plot
                    else:
                        peak_value = np.nan
                        peak_idx = np.nan
                        peak_idx_prom = np.nan
                        missed_interval_count += 1
                        missed_intervals[swing_start:swing_end + 1] = 1

                elif min_max == 'max':
                    peak_idx, _ = find_peaks(z, prominence=0.002)

                    # fig = plt.figure()
                    # plt.plot(z)
                    # plt.plot(peak_idx, z[peak_idx],
                    #          marker='x', linestyle='None', label='clearance_peaks')
                    # plt.show()

                    if len(peak_idx) == 2:
                        peak_value = np.mean(z[peak_idx])  # average of the two max peaks
                        peak_idx_prom = peak_idx[0]  # index for first peak for the prominence plot
                        peak_idx = np.round(np.mean(peak_idx))  # average index for the clearance plot
                    else:
                        peak_value = np.nan
                        peak_idx = np.nan
                        peak_idx_prom = np.nan
                        missed_interval_count += 1
                        missed_intervals[swing_start:swing_end + 1] = 1

                clearance[side].append(peak_value)
                clearance_idx[side].append(peak_idx + swing_start)
                clearance_idx_prom[side].append(peak_idx_prom + swing_start)

            clearance[side] = np.asarray(clearance[side])
            clearance_idx[side] = np.asarray(clearance_idx[side])

            # debugging: check prominence thresholds
            if self.show_figs != 0:
                if min_max == 'min':
                    z_all = np.multiply(-1, np.array(self.trajectories[side]["position_z"]))
                elif min_max == 'max':
                    z_all = np.array(self.trajectories[side]["position_z"])

                clearance_idx_prom[side] = np.asarray(clearance_idx_prom[side])
                clearance_idx_prom_valid = clearance_idx_prom[side][np.logical_not(np.isnan(clearance_idx_prom[side]))].astype(int)
                clearance_prom = peak_prominences(
                    z_all, clearance_idx_prom_valid
                )[0]
                fig = plt.figure()
                plt.scatter(clearance_idx_prom_valid, clearance_prom, c='orange')
                plt.title(f'prominences of {side} {min_max} clearance peaks')
                plt.xlabel('sample number')
                plt.ylabel('prominence')
                if self.show_figs == 1:
                    # plt.show()
                    continue
                elif self.show_figs == 2:
                    plt.savefig(os.path.join(self.save_fig_directory, side + '_clearance_prom_thresholds.png'),
                                bbox_inches='tight')
                    plt.close(fig)

            # plot detected clearances
            if self.show_figs != 0:
                fig, ax = plt.subplots(figsize=(20, 5))
                plt.plot(self.trajectories[side]["position_z"])
                plt.plot(clearance_idx[side], clearance[side],
                         marker='x', linestyle='None', label='clearance_peaks_mean')
                plt.vlines(x=self.gait_events[side]["samples"][stance_begin], ymin=-0.02, ymax=0.1,
                           color='c', label='swing_end')
                plt.vlines(x=self.gait_events[side]["samples"][stance_end], ymin=-0.02, ymax=0.1,
                           color='limegreen', label='swing_begin')
                plt.title(f'Peak detection for {side} {min_max} clearance' +
                          '\n missed intervals: ' + str(missed_interval_count))
                if sum(missed_intervals) > 0:
                    ax.fill_between(range(len(missed_intervals)),
                                    -0.03, 0.11, where=missed_intervals, alpha=0.3)
                    # transform=ax.get_xaxis_transform())
                plt.legend()
                if self.show_figs == 1:
                    # plt.show()
                    continue
                elif self.show_figs == 2:
                    plt.savefig(os.path.join(self.save_fig_directory, side + '_clearance.png'),
                                bbox_inches='tight')
                    plt.close(fig)

        return clearance

    def swing_time(self):
        swing_time = {"left": [], "right": []}
        stance_begin = self.gait_events["stance_begin"]
        for side in swing_time.keys():
            stance_end = self.gait_events["stance_end"]
            swing_time[side] = np.array(
                self.gait_events[side]["times"][stance_begin][1:]
            ) - np.array(self.gait_events[side]["times"][stance_end])
        return swing_time

    def stance_time(self):
        stance_time = {"left": [], "right": []}
        stance_begin = self.gait_events["stance_begin"]
        stance_end = self.gait_events["stance_end"]
        for side in stance_time.keys():
            stance_time[side] = np.array(
                self.gait_events[side]["times"][stance_end]
            ) - np.array(self.gait_events[side]["times"][stance_begin][:-1])
        return stance_time

    def angular_distance(self, q1, q2):
        dot_product = np.dot(q1, q2) 
        # Ensure the dot product is within the valid range for arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        return 2 * np.arccos(dot_product)

    def angle_change(self):
        """
        simple calculation of angle change at the xy plane within one stride
        @return:
        @rtype:
        """
        angles_all = {"left": [], "right": []}
        angle_change = {"left": [], "right": []}
        stance_begin = self.gait_events["stance_begin"]

        for side in angle_change.keys():
            rotation = rot.from_quat(self.trajectories[side][['rotation_w', 'rotation_x', 'rotation_y', 'rotation_z']])
            rot_euler = rotation.as_euler('xyz', degrees=False)
            euler_df = pd.DataFrame(data=rot_euler, columns=['x', 'y', 'z'])
            euler_df['time'] = self.trajectories[side]['time']

            for start, end in zip(
                    self.gait_events[side]["samples"][stance_begin][:-1],
                    self.gait_events[side]["samples"][stance_begin][1:],
            ):
                quat_change = np.array(
                    [
                        self.trajectories[side]["rotation_w"][end],
                        self.trajectories[side]["rotation_x"][end],
                        self.trajectories[side]["rotation_y"][end],
                        self.trajectories[side]["rotation_z"][end],
                    ]
                ) - np.array(
                    [
                        self.trajectories[side]["rotation_w"][start],
                        self.trajectories[side]["rotation_x"][start],
                        self.trajectories[side]["rotation_y"][start],
                        self.trajectories[side]["rotation_z"][start],
                    ]
                )

                # quat_start = np.array([
                # self.trajectories[side]["rotation_w"][start],
                # self.trajectories[side]["rotation_x"][start],
                # self.trajectories[side]["rotation_y"][start],
                # self.trajectories[side]["rotation_z"][start]
                # ])
                
                # quat_end = np.array([
                #     self.trajectories[side]["rotation_w"][end],
                #     self.trajectories[side]["rotation_x"][end],
                #     self.trajectories[side]["rotation_y"][end],
                #     self.trajectories[side]["rotation_z"][end]
                # ])

                # # Normalize quaternions to ensure they are unit quaternions
                # quat_start /= np.linalg.norm(quat_start)
                # quat_end /= np.linalg.norm(quat_end)

                # # Calculate the angular distance between the start and end quaternions
                # orientation_change = self.angular_distance(quat_start, quat_end)

                # #### Debugging: Plot the four dimensions from start to end
                # Extract quaternion values
                quaternions = [
                    [
                        self.trajectories[side]["rotation_w"][i],
                        self.trajectories[side]["rotation_x"][i],
                        self.trajectories[side]["rotation_y"][i],
                        self.trajectories[side]["rotation_z"][i]
                    ] for i in range(start, end + 1)
                ]

                # Normalize quaternions to ensure they are unit quaternions
                quaternions = [q / np.linalg.norm(q) for q in quaternions]

                # Calculate angular distances between consecutive quaternions
                angular_distances = [
                    self.angular_distance(quaternions[i], quaternions[i + 1])
                    for i in range(len(quaternions) - 1)
                ]
                # if max(angular_distances) > 0.2:
                #     time_range = range(start, end + 1)
                #     plt.figure(figsize=(10, 6))

                #     plt.plot(time_range, self.trajectories[side]["rotation_w"][start:end + 1], label='rotation_w')
                #     plt.plot(time_range, self.trajectories[side]["rotation_x"][start:end + 1], label='rotation_x')
                #     plt.plot(time_range, self.trajectories[side]["rotation_y"][start:end + 1], label='rotation_y')
                #     plt.plot(time_range, self.trajectories[side]["rotation_z"][start:end + 1], label='rotation_z')

                #     plt.plot(
                #         time_range[:-1], angular_distances, label="norm", linestyle="--", color="black"
                #     )

                #     plt.xlabel('Time')
                #     plt.ylabel('Rotation')
                #     plt.title(f'Rotation Dimensions from Index {start} to {end}')
                #     plt.legend()
                #     plt.grid(True)
                #     plt.show()

                angle_change[side].append(np.percentile(angular_distances, 75))  # try to use median to detect orientation changes
                # angle_change[side].append(np.linalg.norm(quat_change[0:4]))
                # angle_change[side].append(quat_change[3])  # use quat_z, other axis could also work

        return angle_change
