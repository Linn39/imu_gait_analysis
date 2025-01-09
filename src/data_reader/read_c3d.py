#### read c3d files from Kiel, determine if there is a scaling problem from the mocap data

from cProfile import label
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import c3d
from PyMoCapViewer import MoCapViewer
from scipy.signal import find_peaks, peak_prominences

# joints is a list with the markers of interest

def read_position_data_from_c3d(file_path: str):
    with open(file_path, "rb") as handle:
        reader = c3d.Reader(handle)
        labels = [label.strip() for label in reader.point_labels]
        joints = labels  # quick start, select all available joints/markers
        indices = [labels.index(ele) for ele in joints]

        frames = []
        for i, points, analog in reader.read_frames():
            frames.append(points[indices, :3].reshape(-1))

    columns = [f"{j} ({axis})" for j in joints for axis in ["x", "y", "z"]]
    df = pd.DataFrame(frames, columns=columns)
    df.index = df.index / 200  # Millisecond timestamps

    # interpolate missing values
    df.replace(0, np.nan, inplace=True)
    df.interpolate(method="polynomial", order=2, inplace=True)  # fill missing values
    df.dropna(inplace=True)
    # df.reset_index(drop=True, inplace=True)

    return df

def traj_len(start_x, start_y, start_z, end_x, end_y, end_z):
    """calculate the distance between two points in 3d space

    Args:
        start_x (_type_): _description_
        start_y (_type_): _description_
        start_z (_type_): _description_
        end_x (_type_): _description_
        end_y (_type_): _description_
        end_z (_type_): _description_
    """
    total_traj = np.array(
                    [
                        end_x,
                        end_y,
                        end_z,
                    ]
                ) - np.array(
                    [
                        start_x,
                        start_y,
                        start_z,
                    ]
                )

    total_traj_len = np.linalg.norm(total_traj[0:3])  # calculate using x y z axis (alternatively: only x y)
    # print(f'Trajectory total length {foot[0]} = {total_traj_len}')
    return total_traj_len


#### main ####
if __name__ == '__main__':

    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'path.json')) as f:
        paths = json.loads(f.read())
    data_base_path = paths['data_kiel_mocap']

    file_name = os.path.join(data_base_path, 'pp2_0023.c3d')
    df = read_position_data_from_c3d(os.path.join(data_base_path, file_name))

    # file_name = os.path.join(data_base_path, 'raw', 'pp010', 'treadmill', 'optical', 'optical.csv')# '  # 'pp2_0023.c3d', 'pp169_0007.c3d'
    # df = pd.read_csv(file_name, index_col=0)

    # #### render the 3d visualization
    # render = MoCapViewer(sampling_frequency=200, sphere_radius=0.05)
    # render.add_skeleton(df, skeleton_connection="vicon", color="gray")
    # render.show_window()

    #### plot feet trajectories and start and end markers
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    for foot in ['l', 'r']:
        # plot start and end points used to calculate trajectory lengths
        ax.plot(df[f"{foot}_heel (x)"].iloc[-1], df[f"{foot}_heel (y)"].iloc[-1], df[f"{foot}_heel (z)"].iloc[-1], label=f'{foot}_end', marker='o', markersize=7)
        ax.plot(df[f"{foot}_heel (x)"].iloc[0], df[f"{foot}_heel (y)"].iloc[0], df[f"{foot}_heel (z)"].iloc[0], label=f'{foot}_start', marker='o', markersize=7)

        heel_traj_len = traj_len(
            df[f"{foot}_heel (x)"].iloc[0], 
            df[f"{foot}_heel (y)"].iloc[0], 
            df[f"{foot}_heel (z)"].iloc[0],
            df[f"{foot}_heel (x)"].iloc[-1], 
            df[f"{foot}_heel (y)"].iloc[-1], 
            df[f"{foot}_heel (z)"].iloc[-1],
        )
        ax.plot(df[f"{foot}_heel (x)"], df[f"{foot}_heel (y)"], df[f"{foot}_heel (z)"], label=f'{foot}_heel, total = {round(heel_traj_len/1000, 2)} m')

        # use peak detection to identify the strides
        to_events, _ = find_peaks(-df[f"{foot}_heel (z)"], prominence=50)
        ax.plot(
            df[f"{foot}_heel (x)"].iloc[to_events], 
            df[f"{foot}_heel (y)"].iloc[to_events], 
            df[f"{foot}_heel (z)"].iloc[to_events],
            label='TO events',
            linestyle='None',
            marker='x', markersize=4,
            )
        to_event_prom = peak_prominences(
            -df[f"{foot}_heel (z)"],  to_events
        )[0]

        # calculate stride lengths using the TO events
        to_pos_df = df.iloc[to_events]  # get positions of the to events
        to_pos_diff = to_pos_df[['l_heel (x)', 'l_heel (y)', 'l_heel (z)']].diff()  # get xyz of each stride
        # get each stride length and their timestamps
        print(f'Timestamps: {to_pos_df.index.values}')
        print(f'Stride lengths {foot}: {np.linalg.norm(to_pos_diff, axis=1)}')

    for num in [1, 2]:
        # calculate distance between start and end
        df_markers = df[[f'start_{num} (x)', f'start_{num} (y)', f'start_{num} (z)', f'end_{num} (x)', f'end_{num} (y)', f'end_{num} (z)']].mean()
        marker_arr = [df_markers[f'start_{num} (x)'] - df_markers[f'end_{num} (x)'], df_markers[f'start_{num} (y)'] - df_markers[f'end_{num} (y)'], df_markers[f'start_{num} (z)'] - df_markers[f'end_{num} (z)']]
        print(f"start end marker {num} distance: {np.linalg.norm(marker_arr)}")

        # plot the start and end markers in the trajectory plot
        ax.plot(df[f'start_{num} (x)'], df[f'start_{num} (y)'], df[f'start_{num} (z)'], label=f'start_{num}', marker='o', markersize=7) 
        ax.plot(df[f'end_{num} (x)'], df[f'end_{num} (y)'], df[f'end_{num} (z)'], label=f'end_{num}', marker='o', markersize=7) 

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    # #### plot IMU trajectories
    # imu_data_base_path = paths['data_kiel_val']
    # interim_data_path = os.path.join(imu_data_base_path, 'interim', 'pp002', 'walk_slow') # get cached trajectories
    # if (
    #         os.path.exists(
    #             os.path.join(interim_data_path, "_trajectory_estimation_left.json")
    #         )
    #         and os.path.exists(
    #             os.path.join(interim_data_path, "_trajectory_estimation_right.json")
    #         )
    #         # and not overwrite
    #     ):
    #         print("load interim trajectories")

    #         trajectories = {}
    #         fig2 = plt.figure()
    #         ax2 = fig2.add_subplot(111, projection='3d')
    #         for foot in [("left", "LF"), ("right", "RF")]:
    #             trajectories[foot[0]] = pd.read_json(
    #                 os.path.join(interim_data_path, "_trajectory_estimation_" + foot[0] + ".json")
    #             )
                
    #             # calculate total trajectory distance 
    #             imu_traj_len = traj_len(
    #                 trajectories[foot[0]]["position_x"].iloc[0],
    #                 trajectories[foot[0]]["position_y"].iloc[0],
    #                 trajectories[foot[0]]["position_z"].iloc[0],
    #                 trajectories[foot[0]]["position_x"].iloc[-1],
    #                 trajectories[foot[0]]["position_y"].iloc[-1],
    #                 trajectories[foot[0]]["position_z"].iloc[-1],
    #             )

    #             ax2.plot3D(trajectories[foot[0]]["position_x"],  #[100:2000],
    #                        trajectories[foot[0]]["position_y"],  #[100:2000],
    #                        trajectories[foot[0]]["position_z"],  #[100:2000],
    #                        label=f'{foot[1]} total = {round(imu_traj_len, 2)} m')

    #         ax2.set_xlabel("X")
    #         ax2.set_ylabel("Y")
    #         ax2.set_zlabel("Z")
    #         plt.legend()
    #         plt.show()


