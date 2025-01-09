import json
import os
import sys

from LFRF_parameters.pipeline.data_loader import *
from LFRF_parameters.pipeline.event_detector import (
    TuncaEventDetector,
    LaidigEventDetector,
)
from LFRF_parameters.pipeline.trajectory_estimator import TuncaTrajectoryEstimator
from LFRF_parameters.pipeline.reference_loader import OptogaitReferenceLoader
from LFRF_parameters.pipeline.reference_loader import ZebrisReferenceLoader
from LFRF_parameters.pipeline.reference_loader import OpticalReferenceLoader
from LFRF_parameters.pipeline.pipeline import Pipeline


def execute(sub_list, runs, dataset, data_base_path):
    """
    Executes the Playground pipeline.
    Returns
    -------

    """
    # configure the pipeline
    if dataset == "data_kiel" or dataset == "data_kiel_val":
        pipeline_config = {
            # @name: the name should be unique for each pipeline configuration.
            # it is used to identify interim data and reuse it in the next run
            "name": "data_kiel",
            "raw_base_path": os.path.join(data_base_path, "raw"),
            "interim_base_path": os.path.join(data_base_path, "interim"),
            "processed_base_path": os.path.join(data_base_path, "processed"),
            "overwrite": False,  # overwrite the trajectory estimations
            "show_figures": 0,  # show figures from intermediate steps. 2: figures are saved; 1: figures are shown; 0: no figures plotted
            "location_kws": ["LF", "RF"],
            "data_loader": PhysilogDataLoader,
            "trajectory_estimator": TuncaTrajectoryEstimator,
            "sampling_rate": 200,
            "gait_event_detector": LaidigEventDetector,  # TuncaEventDetector, # LaidigEventDetector,
            "prominence_search_threshold": 0.3,
            "prominence_ic": 0.1,
            "prominence_fo": 0.3,
            "reference_loader": OpticalReferenceLoader,
            "reference_name": "OpticalSystem",
            "dataset": dataset,
            "runs": runs,
            "subjects": sub_list,
        }
    elif dataset == "data_charite":
        pipeline_config = {
            # @name: the name should be unique for each pipeline configuration.
            # it is used to identify interim data and reuse it in the next run
            "name": "data_charite",
            "raw_base_path": os.path.join(data_base_path, "raw"),
            "interim_base_path": os.path.join(data_base_path, "interim"),
            "processed_base_path": os.path.join(data_base_path, "processed"),
            "overwrite": False,  # overwrite the trajectory estimations
            "show_figures": 0,  # show figures from intermediate steps. 2: figures are saved; 1: figures are shown; 0: no figures plotted
            "location_kws": ["LF", "RF"],
            "data_loader": PhysilogDataLoader,
            "trajectory_estimator": TuncaTrajectoryEstimator,
            "sampling_rate": 120,
            "gait_event_detector": TuncaEventDetector,
            "prominence_search_threshold": 0.7,  # 0.7 for normal walking, 0.3 for severly impaired walking
            "prominence_ic": 0.01,
            "prominence_fo": 0.2,
            "reference_loader": OptogaitReferenceLoader,
            "reference_name": "OptoGait",
            "dataset": dataset,
            "runs": runs,
            "subjects": sub_list,
        }
    elif dataset == "data_imu_validation":
        pipeline_config = {
            # @name: the name should be unique for each pipeline configuration.
            # it is used to identify interim data and reuse it in the next run
            "name": "data_imu_validation",
            "raw_base_path": os.path.join(data_base_path, "raw"),
            "interim_base_path": os.path.join(data_base_path, "interim"),
            "processed_base_path": os.path.join(data_base_path, "processed"),
            "overwrite": False,  # overwrite the trajectory estimations
            "show_figures": 0,  # show figures from intermediate steps. 2: figures are saved; 1: figures are shown; 0: no figures plotted
            "location_kws": ["LF", "RF"],
            "data_loader": PhysilogDataLoader,
            "trajectory_estimator": TuncaTrajectoryEstimator,
            "sampling_rate": 120,
            "gait_event_detector": TuncaEventDetector,
            "prominence_search_threshold": 0.7,
            "prominence_ic": 0.01,
            "prominence_fo": 0.01,
            "reference_loader": OptogaitReferenceLoader,
            "reference_name": "OptoGait",
            "dataset": dataset,
            "runs": runs,
            "subjects": sub_list,
        }

    elif dataset == "data_TRIPOD":
        pipeline_config = {
            # @name: the name should be unique for each pipeline configuration.
            # it is used to identify interim data and reuse it in the next run
            "name": "data_TRIPOD",
            "raw_base_path": os.path.join(data_base_path, "raw"),
            "interim_base_path": os.path.join(data_base_path, "interim"),
            "processed_base_path": os.path.join(data_base_path, "processed"),
            "overwrite": False,  # overwrite the trajectory estimations
            "show_figures": 0,  # show figures from intermediate steps. 2: figures are saved; 1: figures are shown; 0: no figures plotted
            "location_kws": ["LF", "RF"],
            "experiment_duration": 120,  # minimal experiment duration in seconds. this is used to cut out only the relevant data
            "data_loader": PhysilogDataLoader,
            "trajectory_estimator": TuncaTrajectoryEstimator,
            "sampling_rate": 128,
            "gait_event_detector": LaidigEventDetector,  # LaidigEventDetector,  #TuncaEventDetector,
            "prominence_search_threshold": 0.7,
            "prominence_ic": 0.01,
            "prominence_fo": 0.03,
            "reference_loader": OptogaitReferenceLoader, # ZebrisReferenceLoader,  # OptogaitReferenceLoader,
            "reference_name": "OptoGait",  # "Zebris",  # "OptoGait",
            "dataset": dataset,
            "runs": runs,
            "subjects": sub_list,
        }

    # create the pipeline
    pipeline = Pipeline(pipeline_config)

    # list of tuples (run number, subject number)
    everything = [
        (x, y)
        for x in range(0, len(pipeline_config["subjects"]))
        for y in range(0, len(pipeline_config["runs"]))
    ]
    # analyze = [(1, 0), (1, 1), (1, 2)]

    analyze = everything

    if dataset == "data_charite":
        pipeline.execute(analyze)  # calculate gait parameters
    elif dataset == "data_kiel" or dataset == "data_kiel_val":
        # pipeline.execute(analyze)   # calculate gait parameters
        # pipeline.execute_validation(analyze)    # validate gait parameters with reference system
        pipeline.execute_plot_all()  # plot all contorl and storke patients from Kiel from saved data points
    else:  # TRIPOD
        # pipeline.execute(analyze)   # calculate gait parameters
        pipeline.execute_validation(
            analyze
        )  # validate gait parameters with reference system
