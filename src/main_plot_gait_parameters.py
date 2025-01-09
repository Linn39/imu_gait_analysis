import os
import json

from visualization.GaitParameterPlot import GaitParameterPlot


## select data
dataset = "data_charite"  # "data_charite"  # "data_kiel"

subjects = None
runs = None
run_name = None
if dataset == "data_charite":
    subjects = [
        "imu0001",
        "imu0002",
        "imu0003",
        "imu0006",
        "imu0007",
        "imu0008",
        "imu0009",
        "imu0011",
        "imu0012",
        "imu0013",
    ]
    runs = ["visit1", "visit2"]
    run_name = "visit"

if dataset == "data_kiel":  # healthy elderly controls
    subjects = [
        "pp010",
        "pp011",
        "pp028",
        "pp079",
        "pp099",
        "pp105",
        "pp106",
        # "pp137",  # very asymmetrical foot trajectories
        # "pp139",  # very asymmetrical foot trajectories
        "pp158",
        "pp165",
    ]
    runs = [
        "treadmill_speed1",  # constant speed 1
        "treadmill_speed2",  # constant speed 2
    ]
    run_name = "treadmill_speed"
plot_window_distribution = (
    False  # if plot from window, check value distributions from the windows
)

with open("path.json") as f:
    paths = json.loads(f.read())
data_base_path = paths[dataset]
interim_base_path = os.path.join(data_base_path, "interim")
processed_base_path = os.path.join(data_base_path, "processed")

gait_param_plot = GaitParameterPlot(
    data_base_path,
    subjects,
    runs,
    run_name,
    drop_turning_interval=True,
)

for subject in subjects:  # subjects:  ["imu0007"]:
    # gait_param_plot.plot_windowed_feature_distribution(subject)
    # gait_param_plot.radar_plot_px(subject, by_window=True, save_fig=False)
    gait_param_plot.radar_plot(subject, by_window=False, save_fig=True)
    # gait_param_plot.radar_plot(subject, by_window=True, save_fig=True)
    # gait_param_plot.boxplot_windows(subject, "speed", save_fig=True)
    # gait_param_plot.scatter_plot_strides(subject, "speed", save_fig=True)
    # gait_param_plot.plot_LR_diff(subject, "clearance", save_fig=True)
