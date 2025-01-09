from turtle import color
import numpy as np
from scipy.signal import find_peaks, peak_prominences
import matplotlib.pyplot as plt

def gyro_threshold_stance(
    imu, stance_magnitude_threshold=0.5, stance_count_threshold=8
):
    """
    As presented by Tunca et al. (https://doi.org/10.3390/s17040825).
    Detect stance phases according to the given gyroscope magnitude, and minimum sample length threshold value.

    Args:
        imu (IMU): IMU object with data
        stance_magnitude_threshold (float): magnitude threshold
        stance_count_threshold (int): grace period in samples

    Returns:
        np.array: Boolean vector indicating stance phases
    """

    gyro = imu.gyro()

    gyro_mag = np.linalg.norm(gyro, axis=1)

    stance = gyro_mag < stance_magnitude_threshold

    stance_count = 0

    for i in range(0, len(stance)):
        if stance[i]:
            if stance_count < stance_count_threshold:
                stance[i] = False
            stance_count += 1
        else:
            if stance_count >= stance_count_threshold:
                for j in range(1, stance_count_threshold + 1):
                    stance[i - j] = False
            stance_count = 0

    # samples = list(range(0, len(gyro_mag)))
    # fig, ax = plt.subplots()
    # ax.fill_between(samples, 0, 1, where=stance, alpha=0.4, transform=ax.get_xaxis_transform())
    #
    # plt.plot(samples, gyro_mag)
    # plt.hlines(stance_magnitude_threshold, 0, len(samples) - 1, colors='orange')
    # plt.title("Gyro Magnitude " + str(stance_magnitude_threshold), fontsize=13)
    # plt.xlabel('Samples')
    # plt.ylabel('Gyroscope Magnitude (rad/s)')
    # plt.show()

    return stance


def tunca_gait_events(imu, stance_magnitude_threshold, stance_count_threshold, prominence_search_threshold, prominence_ic, prominence_fo, show_figs, trajectories, save_fig_directory):
    """
    Gait event detection by Tunca et al. (https://doi.org/10.3390/s17040825)

    Args:
        imu (IMU): IMU object with data
        gyro_threshold (float): gyroscope threshold for identifying stance phases
        stance_count_threshold (int): stance count threshold for identifying stance phases
        show_figs (int): 0: do not create figures, 1: show figures, 2: save figures
        save_fig_directory: directory to save figures

    Returns:
        tuple[list[float], ...]: index of IC_samples, FO_samples, timestamps of IC_times, FO_times and stance indicator
    """

    stance = gyro_threshold_stance(
        imu,
        stance_magnitude_threshold=stance_magnitude_threshold,
        stance_count_threshold=stance_count_threshold,
    )
    step_begins = np.where(np.logical_and(np.logical_not(stance[:-1]), stance[1:]))[0]

    gyro = imu.gyro()
    t = imu.time()

    # select axis with highest std.dev.
    G = np.transpose(gyro)[np.argmax(gyro.std(axis=0))]

    # The signal needs to be inverted to fit the peak detection algorithm.
    # (Flip ot not flip depends on sensor orientation on the shoe.)
    G = -G

    # calculate tilt
    dt = np.diff(t)
    tilt = np.cumsum(G[1:] * dt)

    # recalculate tilt without bias
    tilt_diff_bias = tilt[-1] / (t[-1] - t[0])
    tilt_diff = G - tilt_diff_bias
    tilt = np.cumsum(tilt_diff[1:] * dt)

    # calculate jerk norm for initial contact detection
    jerk_norm = np.linalg.norm(np.gradient(imu.accel(), axis=0), axis=1)

    # get start and end of search regions
    fo_search_threshold = prominence_search_threshold
    ic_search_threshold = prominence_search_threshold
    fo_search_region_ends, _ = find_peaks(-tilt, prominence=fo_search_threshold)
    ic_search_region_starts, _ = find_peaks(tilt, prominence=ic_search_threshold)

    # debugging: check prominence thresholds
    if show_figs != 0:
        fo_search_prom = peak_prominences(
            -tilt,  fo_search_region_ends
        )[0]
        ic_search_prom = peak_prominences(
            tilt,  ic_search_region_starts
        )[0]
        fig = plt.figure()
        plt.scatter(fo_search_region_ends, fo_search_prom,
                    label='fo, n=' + str(len(fo_search_prom)) + ', threshold = ' + str(fo_search_threshold),
                    c='orange')
        plt.scatter(ic_search_region_starts, ic_search_prom,
                    label='ic, n=' + str(len(ic_search_prom)) + ', threshold = ' + str(ic_search_threshold),
                    c='darkturquoise')
        plt.axhline(y=fo_search_threshold, color='coral', linestyle='-')
        plt.axhline(y=ic_search_threshold, color='teal', linestyle='-')
        plt.title('prominences of IC FO search regions')
        plt.xlabel('sample number')
        plt.ylabel('prominence')
        plt.legend()
        if show_figs == 1:
            plt.show()
        elif show_figs == 2:
            plt.savefig(save_fig_directory +  '_IC_FO_search_prom_thresholds.png',
                        bbox_inches='tight')
            plt.close(fig)

    FO_samples = np.array([], dtype=int)
    IC_samples = np.array([], dtype=int)
    FO_prominences = np.array([])
    IC_prominences = np.array([])
    missed_interval_count = 0
    atypical_interval_count = 0
    missed_intervals = np.zeros(len(tilt_diff), dtype=bool)
    atypical_intervals = np.zeros(len(tilt_diff), dtype=bool)
    fo_prom_threshold = prominence_fo  # 1.5
    ic_prom_threshold = prominence_ic  # 0.1

    for step_begin, step_end in zip(step_begins[:-1], step_begins[1:]):

        # identify search region boundaries within one step
        fo_search_region_end = fo_search_region_ends[
            np.logical_and(
                fo_search_region_ends > step_begin, fo_search_region_ends < step_end
            )
        ]
        ic_search_region_start = ic_search_region_starts[
            np.logical_and(
                ic_search_region_starts > step_begin,
                ic_search_region_starts <= step_end,
            )
        ]

        # for healthy gait: there shoud be exactly one FO and one IC search region boundary and the search regions should not overlap,
        # otherwise events cannot be detected properly
        if (
            len(fo_search_region_end) == 1
            and len(ic_search_region_start) == 1
            and ic_search_region_start >= fo_search_region_end
        ):
            fo_search_region_end = fo_search_region_end[0]
            ic_search_region_start = ic_search_region_start[0]

            fo_cand, _ = find_peaks(
            -tilt_diff[step_begin:fo_search_region_end], prominence=fo_prom_threshold
            )
            ic_cand, _ = find_peaks(
                -tilt_diff[ic_search_region_start:step_end], prominence=ic_prom_threshold
            )
            fo_prominence_cand = peak_prominences(
                -tilt_diff[step_begin:fo_search_region_end], fo_cand
            )[0]
            ic_prominence_cand = peak_prominences(
                -tilt_diff[ic_search_region_start:step_end], ic_cand
            )[0]

            # transform step-local indices of the candidates to full signal indices
            fo_cand = np.asarray([x + step_begin for x in fo_cand])
            ic_cand = np.asarray([x + ic_search_region_start for x in ic_cand])

            # for healthy gait: identify the first event before and after the search region boundary
            if np.any(
                np.logical_and(fo_cand < fo_search_region_end, fo_cand >= step_begin)
            ) and np.any(
                np.logical_and(ic_cand > ic_search_region_start, ic_cand < step_end)
            ) and fo_cand[fo_cand < fo_search_region_end][-1] < ic_cand[ic_cand > ic_search_region_start][0]:  # fo has to occur before ic

                FO_samples = np.append(
                    FO_samples,
                    fo_cand[fo_cand < fo_search_region_end][-1],
                )
                IC_samples = np.append(
                    IC_samples,
                    ic_cand[ic_cand > ic_search_region_start][0],
                )
                FO_prominences = np.append(
                    FO_prominences,
                    fo_prominence_cand[fo_cand < fo_search_region_end][-1],
                )
                IC_prominences = np.append(
                    IC_prominences,
                    ic_prominence_cand[ic_cand > ic_search_region_start][0],
                )
            else:
                # No ic or fo found in search region
                missed_interval_count += 1
                missed_intervals[step_begin:step_end + 1] = 1
                # continue

        elif (
            len(fo_search_region_end) == 1
        ):
            # for pathological gait, if there is no toe-lift (i.e., IC search region start cannot be detected), 
            fo_search_region_end = fo_search_region_end[0]
            fo_cand, _ = find_peaks(
            -tilt_diff[step_begin:fo_search_region_end], prominence=fo_prom_threshold
            )
            fo_prominence_cand = peak_prominences(
                -tilt_diff[step_begin:fo_search_region_end], fo_cand
            )[0]

            # #### use smallest vertical foot position to find the IC event instead (as explained in Tunca et al. 2017)
            # # get index of the IC event from smallest foot vertical position
            # ic_cand = np.argmin(trajectories["velocity_z"].values[step_begin:step_end])
            # ic_prominence_cand = np.nan  # prominence is not relevant, since IC is not detected by peak detection

            # # get index of the IC event from vertival velocity peaks
            # ic_cand, _ = find_peaks(
            #     (-10) * trajectories["velocity_z"].values[step_begin:step_end], prominence=ic_prom_threshold
            # )
            # ic_prominence_cand = peak_prominences(
            #     (-10) * trajectories["velocity_z"].values[step_begin:step_end], ic_cand
            # )[0]

            # #### get index of the IC event from derivative of the vertical positions (smoothened)
            # ic_cand, _ = find_peaks(
            #     (-1000) * np.gradient(trajectories["position_z"].values[step_begin:step_end]), 
            #     prominence=ic_prom_threshold
            #     )
            # ic_prominence_cand = peak_prominences(
            #     (-1000) * np.gradient(trajectories["position_z"].values[step_begin:step_end]), ic_cand
            # )[0]

            #### use jerk norm threshold (as described in Laidig et al. 2021)
            jerk_threshold = 0.95  # relative to the largers peak value
            jerk_max = np.amax(jerk_norm[fo_search_region_end:step_end])
            # jark_max_idx = jerk.argmax[0]
            ic_cand = np.where(np.diff(np.sign(jerk_norm[fo_search_region_end:step_end] - jerk_threshold*jerk_max)))[0][0]
            ic_cand = [int(ic_cand)]  # convert to list to match other algorithm outputs
            ic_prominence_cand = [np.nan]  # peak prominence not relevant here

            # transform step-local indices of the candidates to full signal indices
            fo_cand = np.asarray([x + step_begin for x in fo_cand])
            # ic_cand = np.asarray([x + step_begin for x in ic_cand])
            ic_cand = np.asarray([x + fo_search_region_end for x in ic_cand])  # for the jerk algorithm

            # ic_cand = np.asarray([ic_cand + step_begin])  # np.argmin already returns the first smalles value, theres only one element

            # for pathological gait: identify the first event before and after the search region boundary
            if np.any(
                np.logical_and(fo_cand < fo_search_region_end, fo_cand >= step_begin)
            ) and np.any(
                np.logical_and(ic_cand > step_begin, ic_cand < step_end)
            ) and fo_cand[fo_cand < fo_search_region_end][-1] < ic_cand[ic_cand < step_end][-1]:  # fo has to occur before ic
                FO_samples = np.append(
                    FO_samples,
                    fo_cand[fo_cand < fo_search_region_end][-1],
                )
                IC_samples = np.append(
                    IC_samples,
                    ic_cand[ic_cand < step_end][-1],
                )
                FO_prominences = np.append(
                    FO_prominences,
                    fo_prominence_cand[fo_cand < fo_search_region_end][-1],
                )
                IC_prominences = np.append(
                    IC_prominences,
                    ic_prominence_cand,
                    # ic_prominence_cand[ic_cand < step_end][-1]
                )

                atypical_interval_count += 1
                atypical_intervals[step_begin:step_end + 1] = 1

            else:
                # No ic or fo found in search region
                missed_interval_count += 1
                missed_intervals[step_begin:step_end + 1] = 1
                # continue

        else:
            # overlaping or unconclusive search regions
            # if more than one boundary detected: ignore all events for this step
            missed_interval_count += 1
            missed_intervals[step_begin:step_end + 1] = 1
            continue

        # # transform step-local indices of the candidates to full signal indices
        # fo_cand = np.asarray([x + step_begin for x in fo_cand])
        # ic_cand = np.asarray([x + ic_search_region_start for x in ic_cand])

    IC_times = [t[sample] if not np.isnan(sample) else np.nan for sample in IC_samples]
    FO_times = [t[sample] if not np.isnan(sample) else np.nan for sample in FO_samples]

    # plot gait event prominences to confirm thresholds
    if show_figs != 0:
        fig = plt.figure()
        plt.scatter(FO_samples, FO_prominences,
                    label='fo, n=' + str(len(FO_prominences)) + ', threshold = ' + str(fo_prom_threshold),
                    c='orange')
        plt.scatter(IC_samples, IC_prominences,
                    label='ic, n=' + str(len(IC_prominences)) + ', threshold = ' + str(ic_prom_threshold),
                    c='darkturquoise')
        plt.axhline(y=fo_prom_threshold, color='coral', linestyle='-')
        plt.axhline(y=ic_prom_threshold, color='teal', linestyle='-')
        plt.title('prominences from detected IC FO events')
        plt.xlabel('sample number')
        plt.ylabel('prominence')
        plt.legend()
        if show_figs == 1:
            plt.show()
        elif show_figs == 2:
            plt.savefig(save_fig_directory + '_prominence_thresholds.png',
                        bbox_inches='tight')
            plt.close(fig)

    if show_figs != 0:
        # plot gait events
        fig, ax = plt.subplots(figsize=(20, 5))

        # debugging: add acceleration in the plot
        # imu.acc_to_g()  # --> careful! this will modify the imu object, thus leading to wrongly scaled acc data for further processing!
        # acc = imu.accel_mag()
        # plt.plot(range(len(acc)), acc, label="acc_py")

        plt.plot(range(len(G)), G, label="gyro_py")
        plt.plot(range(len(tilt)), tilt, label='tilt')
        plt.plot(range(len(trajectories["position_z"].values)), trajectories["position_z"].values * 10, label='vertical_pos')
        plt.plot(jerk_norm / 10, label="jerk_grad")
        plt.plot(imu.accel_mag() / 10, label="accel_mag")
        # plt.plot(range(len(np.gradient(trajectories["position_z"].values))), np.gradient(trajectories["position_z"].values) * 1000, label='vertical_derivative')
        # plt.plot(range(len(trajectories["velocity_z"].values)), trajectories["velocity_z"].values * 10, label='vertical_velocity')
        plt.vlines(x=fo_search_region_ends, ymin=-10, ymax=9,
                   color='c', label='fo_search_region_ends')
        plt.vlines(x=ic_search_region_starts, ymin=-10, ymax=9,
                   color='m', label='ic_search_region_starts')
        plt.vlines(x=step_begins, ymin=-10, ymax=9,
                   color='y', label='stance_begin')
        # plt.vlines(x=ic_search_region_starts, color='c')
        plt.plot(FO_samples, np.array(G)[FO_samples],
                 marker='x', linestyle='None', label="FO_py")
        plt.plot(IC_samples, np.array(G)[IC_samples],
                 marker='x', linestyle='None', label="IC_py")
        if sum(missed_intervals) > 0:
            ax.fill_between(range(len(missed_intervals)),
                            -12, 12, where=missed_intervals, color="orange", alpha=0.2)
        if sum(atypical_intervals) > 0:
            ax.fill_between(range(len(atypical_intervals)),
                            -11, 13, where=atypical_intervals, color="darkturquoise", alpha=0.1)
        plt.title('IC: ' + str(len(IC_times)) + ', FO: ' + str(len(FO_times)) +
                  '\n' + str(atypical_interval_count) + ' atypical intervals' + 
                  '\n missed ' + str(missed_interval_count) + ' search intervals'
                  )
        plt.xlabel('Time (s)')
        plt.legend()
        if show_figs == 1:
            plt.show()
        elif show_figs == 2:
            plt.savefig(save_fig_directory + '_gait_events.png',
                        bbox_inches='tight')
            plt.close(fig)

    return IC_samples, FO_samples, IC_times, FO_times, stance


def zero_crossing(x_1, x_2, y_1, y_2):
    """ linear interpolation for zero crossing between points (x_1, y_1) and (x_2, y_2) """
    
    return -((y_1*x_2-y_2*x_1)/(x_2-x_1))/((y_2-y_1)/(x_2-x_1))


def hundza_gait_events(imu):
    """ gait event detection by Hundza et al. (https://doi.org/10.1109/TNSRE.2013.2282080) """ 

    gyro = imu.gyro()
    t = imu.time()
    
    # select axis with highest std.dev.
    G = np.transpose(gyro)[np.argmax(gyro.std(axis=0))]

    # since left and right foot sensors are mirrored, the signal needs to be inverted.
    if len(G[G > 0]) > len(G[G <= 0]):
        G = -G

    # find local minima and maxima
    maxima, _ = find_peaks(G, prominence=1)
    minima, _ = find_peaks(-G, prominence=1)

    # filter maxima for maxima above a threshold
    threshold = 3
    positive_maxima = np.intersect1d(maxima, np.argwhere(G > threshold))

    TOFS = np.array([])   # Termination of forward swing
    IOFS = np.array([])   # Initiation of forwars swing
    TO = np.array([])     # Toe off

    stance = np.ones_like(t, dtype=bool)

    # indices of all zero or negative samples
    negative_zero_idx = np.asarray((G <= 0).nonzero())
    
    for maximum_id in positive_maxima:
        # find first zero crossing AFTER the maximum
        potential_tofs = negative_zero_idx[negative_zero_idx > maximum_id][0]
        if G[potential_tofs] == 0: 
            TOFS = np.append(TOFS, t[potential_tofs])
            stance_begin = potential_tofs
        else:
            # if there is no sample at the zero crossing, interpolate linearly to find exact zero crossing time
            tofs = zero_crossing(t[potential_tofs - 1], t[potential_tofs], G[potential_tofs - 1], G[potential_tofs])
            TOFS = np.append(TOFS, tofs)
            if tofs - t[potential_tofs - 1] < t[potential_tofs] - tofs:
                stance_begin = potential_tofs - 1
            else:
                stance_begin = potential_tofs

        # find first zero crossing BEFORE the maximum
        potential_iofs = negative_zero_idx[negative_zero_idx < maximum_id][-1]
        if G[potential_iofs] == 0:
            IOFS = np.append(IOFS, t[potential_iofs])
        else:
            # if there is no sample at the zero crossing, interpolate linearly to find exact zero crossing time
            iofs = zero_crossing(t[potential_iofs], t[potential_iofs + 1], G[potential_iofs], G[potential_iofs + 1])
            IOFS = np.append(IOFS, iofs)

        # find the first minimum BEFORE the initiation of forward swing
        to = minima[minima < np.argwhere(t < iofs)[-1]][-1]
        TO = np.append(TO, t[to])

        stance[to:stance_begin] = False

    return TOFS, IOFS, TO, stance


def laidig_gait_events(imu, stance_magnitude_threshold, stance_count_threshold, show_figs, save_fig_directory):
    """
    gait event detection by Laidig et al. (https://doi.org/10.3389/fdgth.2021.736418)
    """

    stance = gyro_threshold_stance(
        imu,
        stance_magnitude_threshold=stance_magnitude_threshold,
        stance_count_threshold=stance_count_threshold,
    )
    # step_begins = np.where(np.logical_and(np.logical_not(stance[:-1]), stance[1:]))[0]
    stance_ends = np.where(np.logical_and(np.logical_not(stance[1:]), stance[:-1]))[0]  # find all ends of stance phase

    gyro = imu.gyro()
    acc = imu.accel()
    t = imu.time()

    # select axis with highest standard deviation for gyro signal
    G = np.transpose(gyro)[np.argmax(gyro.std(axis=0))]

    # fig, ax = plt.subplots(figsize=(20, 5))
    # plt.plot(range(len(G)), G, label="gyro_axis")
    # plt.vlines(x=stance_ends, ymin=-10, ymax=9, label='stance_end')
    # plt.show()

    FO_samples = np.array([], dtype=int)
    IC_samples = np.array([], dtype=int)
    fo_search_begins = np.array([], dtype=int)
    fo_search_ends = np.array([], dtype=int)
    gyro_maxes = np.array([], dtype=int)

    for step_begin, step_end in zip(stance_ends[:-1], stance_ends[1:]):
        current_gyro = G[step_begin:step_end]  # get current stride
        current_gyro = current_gyro * np.sign(sum(current_gyro[0:int(len(current_gyro)/4)]))  # check sensor orientation.
        # first 1/4 of the signal should be positive

        # detect FO events using zero-crossing
        initial_gyro = current_gyro[0:int(len(current_gyro)/3)]  # get the first 1/3 of the gyro signal
        gyro_max_idx = np.argmax(initial_gyro) 
        half_max = np.max(initial_gyro)/2  # get half max value in the initial gyro signal
        fo_search_begin = np.argmax(current_gyro > half_max)  # get index of the half max: start of zero crossing search region
        fo_search_end = np.argmin(current_gyro)
        try:
            FO_local = np.argmax(current_gyro[fo_search_begin:fo_search_end] < 0) + fo_search_begin  # local index of FO event
            FO_samples = np.append(FO_samples, FO_local + step_begin)
            fo_search_begins = np.append(fo_search_begins, fo_search_begin + step_begin)
            fo_search_ends = np.append(fo_search_ends, fo_search_end + step_begin)
            gyro_maxes = np.append(gyro_maxes, gyro_max_idx + step_begin)

            # fig, ax = plt.subplots(figsize=(20, 5))
            # plt.plot(range(len(current_gyro)), current_gyro, label="current_gyro")
            # plt.plot(fo_search_begin, current_gyro[fo_search_begin],
            #             marker='x', linestyle='None', label="half_max")
            # plt.plot(FO_local, current_gyro[FO_local],
            #             marker='x', linestyle='None', label="FO")
            # plt.xlabel('Samples')
            # plt.legend()
            # plt.show()

            # detect IC events using jerk from acceleraion
            # calculate jerk norm for initial contact detection
            jerk_norm = np.linalg.norm(np.gradient(acc, axis=0), axis=1)
            # current_jerk = jerk_norm[(step_begin + FO_local):step_end]  # current search region starts with FO event
            current_jerk = jerk_norm[
                (step_begin + fo_search_end) : step_end
            ]  # current search region starts with the end of the FO search region

            jerk_threshold = 0.95  # 0.95
            jerk_max = np.max(current_jerk)
            IC_local = np.argmax(current_jerk > jerk_threshold * jerk_max)
            # IC_samples = np.append(IC_samples, step_begin + FO_local + IC_local)
            IC_samples = np.append(IC_samples, step_begin + fo_search_end + IC_local)
        except ValueError:  # if FO cannot be detected, skip this gait cycle
            pass

    IC_times = [t[sample] if not np.isnan(sample) else np.nan for sample in IC_samples]
    FO_times = [t[sample] if not np.isnan(sample) else np.nan for sample in FO_samples]

    if show_figs != 0:
        fig, ax = plt.subplots(figsize=(20, 5))
        plt.plot(range(len(G)), G, label="gyro")
        plt.plot(range(len(jerk_norm)), jerk_norm, label="jerk")
        plt.vlines(x=fo_search_begins, ymin=-10, ymax=9, 
                   color="darkturquoise", label='fo_search_begin')
        plt.vlines(x=fo_search_ends, ymin=-10, ymax=9, 
                   color="yellow", label='fo_search_end')
        plt.plot(gyro_maxes, np.array(G)[gyro_maxes],
                    marker='x', linestyle='None', label="gyro_max")
        plt.plot(FO_samples, np.array(G)[FO_samples],
                    marker='x', linestyle='None', label="FO_py")
        plt.plot(IC_samples, np.array(jerk_norm)[IC_samples],
                    marker='x', linestyle='None', label="IC_py")
        plt.xlabel('Samples')
        plt.legend()
        if show_figs == 1:
            plt.show()
        elif show_figs == 2:
            plt.savefig(save_fig_directory + '_gait_events.png',
                        bbox_inches='tight')
            plt.close(fig)

    return IC_samples, FO_samples, IC_times, FO_times, stance
