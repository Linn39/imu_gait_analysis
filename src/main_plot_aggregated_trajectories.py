#### plot aggregated trajectories from all strides ####

#### imports ####
from visualization.FootTrajectoryPlot import FootTrajectoryPlot


## select data
dataset = "data_charite"  # "data_charite"  # "data_kiel"
beautify = (
    True  # remove outliers from the plot (not really marking outlier in the data)
)

subjects = None
runs = None
if dataset == "data_charite":
    subjects = [
        # "imu0001",
        # "imu0002",
        # "imu0003",
        "imu0006",
        # "imu0007",
        # "imu0008",
        # "imu0009",
        # "imu0011",
        # "imu0012",
        # "imu0013",
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
        "pp137",
        "pp139",
        "pp158",
        "pp165",
    ]
    runs = [
        "treadmill_speed1",  # constant speed 1
        "treadmill_speed2",  # constant speed 2
    ]

for subject in subjects:
    foot_trajectory_plot = FootTrajectoryPlot(
        dataset, subject, runs, label_paretic_side=False
    )
    foot_trajectory_plot.plot_aggregated_trajectories(beautify=True)
