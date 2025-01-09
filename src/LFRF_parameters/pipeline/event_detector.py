import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LFRF_parameters.pipeline.abstract_pipeline_components import AbstractEventDetector
from LFRF_parameters.event_detection.imu_event_detection import hundza_gait_events, laidig_gait_events, tunca_gait_events

class TuncaEventDetector(AbstractEventDetector):
    """
    Gait event detection as presented by Tunca et al. (https://doi.org/10.3390/s17040825).
    The actual algorithm is implemented in event_datection/imu_event_detection.py
    """

    def detect(self,
               stance_thresholds,
               interim_base_path,
               dataset,
               subject,
               run,
               prominence_search_threshold, 
               prominence_ic, 
               prominence_fo,
               show_figs,
               trajectories,
               save_fig_directory
               ):
        """
        Detect gait events.

        Args:
            stance_thresholds (dict[str, float]): Gyroscope magnitude and stance count thresholds for stance detection

        Returns:
            dict[str, dict]: IC and FO samples and timestamps for the right and left foot.
        """
        result = {}

        result["stance_begin"] = "IC"
        result["stance_end"] = "FO"

        for foot in [("left", "LF"), ("right", "RF")]:

            IC_samples, FO_samples, IC_times, FO_times, stance = tunca_gait_events(
                self.imus[foot[1]],
                float(stance_thresholds["stance_magnitude_threshold_" + foot[0]]),
                int(stance_thresholds["stance_count_threshold_" + foot[0]]),
                prominence_search_threshold, 
                prominence_ic, 
                prominence_fo,
                show_figs,
                trajectories[foot[0]],
                os.path.join(save_fig_directory, foot[1])
            )

            result[foot[0]] = {
                "samples": {"IC": IC_samples, "FO": FO_samples},
                "times": {"IC": IC_times, "FO": FO_times},
            }

        return result

# class TuncaEventDetector(AbstractEventDetector):
#
#     def detect(self):
#         result = {}
#
#         result["stance_begin"] = "IC"
#         result["stance_end"] = "FO"
#
#         for foot in [("right", "RF"), ("left", "LF")]:
#             IC_samples, FO_samples, IC_times, FO_times, stance = tunca_gait_events(self.imus[foot[1]])
#             plt.plot(self.imus[foot[1]].time(), np.transpose(self.imus[foot[1]].accel())[0])
#             plt.plot(self.imus[foot[1]].time(), np.transpose(self.imus[foot[1]].accel())[1])
#             plt.plot(self.imus[foot[1]].time(), np.transpose(self.imus[foot[1]].accel())[2])
#
#             plt.plot(IC_times, np.zeros_like(IC_times), "xk")
#             plt.plot(FO_times, np.zeros_like(FO_times), "xr")
#             plt.show()
#             result[foot[0]] = {"samples" : {"IC" : IC_samples, "FO" : FO_samples},
#                                "times" : {"IC" : IC_times, "FO" : FO_times}}
#
#         return result

class HundzaEventDetector(AbstractEventDetector):

    def detect(self):
        result = {}

        for foot in [("right", "RL"), ("left", "LL")]:
            TOFS, IOFS, TO, stance = hundza_gait_events(self.imus[foot[1]])
            result[foot[0]] = pd.DataFrame(data={"TOFS": TOFS, "IOFS": IOFS, "TO": TO})

        return result


class LaidigEventDetector(AbstractEventDetector):
    """
    Detect gait events.

    Args:
        stance_thresholds (dict[str, float]): Gyroscope magnitude and stance count thresholds for stance detection

    Returns:
        dict[str, dict]: IC and FO samples and timestamps for the right and left foot.
    """
    def detect(self, 
               stance_thresholds,
               interim_base_path,
               dataset,
               subject,
               run,
               prominence_search_threshold, 
               prominence_ic, 
               prominence_fo,
               show_figs,
               trajectories,
               save_fig_directory
    ):

        result = {}
        result["stance_begin"] = "IC"
        result["stance_end"] = "FO"
        
        for foot in [("left", "LF"), ("right", "RF")]:

            IC_samples, FO_samples, IC_times, FO_times, stance = laidig_gait_events(
                self.imus[foot[1]],
                float(stance_thresholds["stance_magnitude_threshold_" + foot[0]]),
                int(stance_thresholds["stance_count_threshold_" + foot[0]]),
                show_figs,
                os.path.join(save_fig_directory, foot[1])
            )

            result[foot[0]] = {
                "samples": {"IC": IC_samples, "FO": FO_samples},
                "times": {"IC": IC_times, "FO": FO_times},
            }

        return result