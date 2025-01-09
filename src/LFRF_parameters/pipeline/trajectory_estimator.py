""" This module contains the implementation of a trajectory estimator. """

import matplotlib.pyplot as plt
import pandas as pd
import os

from .abstract_pipeline_components import AbstractTrajectoryEstimator
from LFRF_parameters.trajectory_estimation.filter import error_state_kalman_filter


class TuncaTrajectoryEstimator(AbstractTrajectoryEstimator):
    """
    Trajectory estimator based on Tunca et al. (https://doi.org/10.3390/s17040825).
    The actual error-state Kalman filter is implemented in trajectory_estimation/filter.py
    """

    def estimate(
        self,
        # name,
        interim_base_path,
        dataset,
        subject,
        run,
        imu_ic,
        stance_thresholds,
        sampling_rate,
        overwrite,
        show_figs,
        save_fig_directory
    ):
        """
        Estimate trajectories from IMU data.

        Args:
            name (str): Name of the configuration (used to identify cache files)
            interim_base_path (str): Folder where caching data can be stored
            dataset (str): Folder containing the dataset
            subject (str): Identifier of the subject
            run (str): Identifier of the run
            imu_ic (float): timestamp of initial contact in the IMU data
            stance_thresholds (dict[str, float]): Gyroscope magnitude and stance count thresholds for stance detection for the right and left foot
            overwrite (bool): Flag if cached files should be overwritten

        Returns:
            dict[str, DataFrame]: DataFrames with trajectory information for the right and left foot
        """

        interim_data_path = os.path.join(interim_base_path, subject, run)  # dataset

        # check if cached results are present
        if (
            os.path.exists(
                os.path.join(interim_data_path, "_trajectory_estimation_left.json")
            )
            and os.path.exists(
                os.path.join(interim_data_path, "_trajectory_estimation_right.json")
            )
            and not overwrite
        ):
            print("load interim trajectories")

            trajectories = {}
            for foot in [("left", "LF"), ("right", "RF")]:
                trajectories[foot[0]] = pd.read_json(
                    os.path.join(interim_data_path, "_trajectory_estimation_" + foot[0] + ".json")
                )

            # return {
            #     "left": pd.read_json(
            #         os.path.join(
            #             interim_data_path, "_trajectory_estimation_left.json"
            #         )
            #     ),
            #     "right": pd.read_json(
            #         os.path.join(
            #             interim_data_path, "_trajectory_estimation_right.json"
            #         )
            #     ),
            # }
        else:
            # calculate trajectories
            os.makedirs(interim_data_path, exist_ok=True)
            trajectories = {}

            for foot in [("left", "LF"), ("right", "RF")]:
                # "LF" and "RF" correspond to the filenames of the respective sensors
                trajectory = error_state_kalman_filter(
                    self.imus[foot[1]],
                    sampling_rate,
                    imu_ic,
                    zero_z=True,
                    zero_xz=False,
                    stance_magnitude_threshold=float(
                        stance_thresholds["stance_magnitude_threshold_" + foot[0]]
                    ),
                    stance_count_threshold=int(
                        stance_thresholds["stance_count_threshold_" + foot[0]]
                    ),
                )
                # cache results
                trajectory.to_json(
                    os.path.join(
                        interim_data_path,
                        "_trajectory_estimation_" + foot[0] + ".json",
                    )
                )

                trajectories[foot[0]] = trajectory

        # plot loaded / calculated trajectories
        if show_figs != 0:
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            for foot in [("left", "LF"), ("right", "RF")]:
                ax.plot3D(trajectories[foot[0]]["position_x"],  #[100:2000],
                          trajectories[foot[0]]["position_y"],  #[100:2000],
                          trajectories[foot[0]]["position_z"],  #[100:2000],
                          label=foot[1])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.legend()
            if show_figs == 1:
                plt.show()
            elif show_figs == 2:
                plt.savefig(os.path.join(save_fig_directory, foot[1] + '_3D_trajectories.png'),
                            bbox_inches='tight')
                plt.close(fig)

        return trajectories
