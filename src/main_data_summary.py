# summarize participant and sensor data

from data_reader.SubjectInfo import SubjectInfo
from data_reader.ImuDataSummary import ImuDataSummary

## Kiel dataset ##
control_list = [
    "pp010",
    "pp011",
    "pp028",
    # "pp071",    # part of the RF data is not usable (IMU dropped from the foot)
    "pp079",
    "pp099",
    "pp105",
    "pp106",
    # "pp114",    # only one constant treadmill speed
    "pp137",
    "pp139",
    "pp158",
    "pp165",
    # "pp166"     # only one constant treadmill speed
]

stroke_list = [
    "pp077",
    "pp101",
    "pp109",
    "pp112",
    "pp122",
    "pp123",
    "pp145",
    "pp149",
]

overground_list = [
    # "pp001",
    "pp002",
    "pp003",
    "pp004",
    "pp005",
    "pp006",
    # "pp007",    # l_psis is missing for the optical data
    "pp008",
    # "pp009",
    # "pp010"
]

charite_list = [
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

# summarize participant data
subject_info = SubjectInfo(dataset="charite", sub_list=charite_list)
subject_info.anthropometrics()
subject_info.improvement_evaluation()

# summarize sensor data
imu_data_summary = ImuDataSummary(
    dataset="charite",
    sub_list=charite_list,
    run_list=["visit1", "visit2"],
    parameter_list=[
        "stride_length_avg",
        "speed_avg",
        "stride_length_CV",
        "speed_CV",
        "stride_length_SI",
        "speed_SI",
    ],
)

imu_data_summary.check_imu_files(
    location_list=["LF", "RF", "LW", "RW", "SA"]
)  # check if all IMU files exist
imu_data_summary.imu_raw_data_summary(
    imu_locations=["LF", "SA"]
)  # summarize raw IMU data
imu_data_summary.gait_params_summary()  # summarize gait parameters
