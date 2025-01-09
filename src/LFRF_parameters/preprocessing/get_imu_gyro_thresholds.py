"""
===========
Slider for gyro stance threshold
===========

"""
from curses import raw
import sys,os
print(os.getcwd())
sys.path.append(os.getcwd())
sys.path.append("./src")
import csv
import json
import numpy as np
import pandas as pd
import scipy as sp
from data_reader.imu import IMU
from data_reader.DataLoader import DataLoader 
from LFRF_parameters.event_detection.imu_event_detection import gyro_threshold_stance
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class GyroPlot():
    def __init__(self, read_path, save_path, sub, run):
        self.read_path = read_path
        self.save_path = save_path
        self.sub = sub
        self.run = run

    def update(self, val):
        gyro_threshold = self.sgyro_mag.val
        count = int(self.scount.val)
        stance = gyro_threshold_stance(
            self.imu,
            stance_magnitude_threshold=gyro_threshold,
            stance_count_threshold=count,
        )
        self.hline.set_ydata(gyro_threshold)
        # delete old stance_shadows for re-plot
        for collection in self.ax.collections:
            if str(collection.get_label()) == "Stance":
                collection.remove()
        self.stance_shadows = self.ax.fill_between(self.samples, 0, 1, where=stance, alpha=0.4,
                                        facecolor='skyblue',
                                        transform=self.ax.get_xaxis_transform(), label='Stance')
        self.fig.canvas.draw_idle()


    def reset(self, event):
        self.scount.reset()
        self.sgyro_mag.reset()


    def save(self, event):
        self.stance_magnitude_thresholds[self.foot] = self.sgyro_mag.val
        self.stance_count_thresholds[self.foot] = self.scount.val

        plt.savefig(os.path.join(self.save_path,
                                str('stance_threshold_' + self.foot + '_.png')),
                    bbox_inches='tight')


    def change_color(self, event):
        self.button_save.color = '0.7'


    def check_duplicates(self, file_path):
        df = pd.read_csv(file_path)
        dup = df.groupby(['subject', 'run']).size() > 1
        if dup.any():
            print('!!!!!!!!!! duplicate entries !!!!!!!!!! see below:')
            print(dup[dup != 0])
        else:
            print("======== No duplicate entries for gyro stance threshold. ========")

    def gyro_threshold_slider(self):
        c0 = 8  # initial stance count threshold
        g0 = 0.7  # initial gyro magnitude threshold
        # setup
        delta_c = 1  # stance count resolution
        delta_g = 0.1  # gyro magnitude resolution

        # get the data file
        gyro_thresholds = []  # save one subject at a time
        self.stance_magnitude_thresholds = {"LF": None, "RF": None}
        self.stance_count_thresholds = {"LF": None, "RF": None}

        for self.foot in ["LF", "RF"]:
            imu_path = os.path.join(self.read_path, self.foot + ".csv")
            self.imu = IMU(imu_path)
            self.imu.gyro_to_rad()
            gyro_mag = np.linalg.norm(self.imu.gyro(), axis=1)
            self.samples = np.arange(len(gyro_mag))

            stance = gyro_threshold_stance(
                self.imu,
                stance_magnitude_threshold=g0,
                stance_count_threshold=c0,
            ).astype(bool)

            # make plot
            self.fig, self.ax = plt.subplots(figsize=(12, 5))
            plt.subplots_adjust(left=0.12, bottom=0.3)
            l, = plt.plot(self.samples, gyro_mag, lw=1)
            self.hline = self.ax.axhline(y=g0, xmin=0.0, xmax=1.0, color='coral', label='Gyro Magnitude Threshold')
            self.stance_shadows = self.ax.fill_between(self.samples, 0, 1, where=stance, alpha=0.4,
                                                facecolor='skyblue',
                                                transform=self.ax.get_xaxis_transform(), label='Stance')
            plt.title('Manual Stance Detection' +
                        '\n' + self.sub + '  ' + self.run + '  ' + self.foot)
            plt.legend()
            plt.ylabel('Gyro Magnitude (rad/s)')
            plt.xlabel('Sample Number')
            self.ax.margins(x=0)

            axcolor = '0.9'
            axcount = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
            axgyro_mag = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

            self.scount = Slider(axcount, 'Count Threshold', 3, 30, valinit=c0, valstep=delta_c)
            self.sgyro_mag = Slider(axgyro_mag, 'Gyro Mag. Threshold', 0.1, 2.5, valinit=g0, valstep=delta_g)
            # The vline attribute controls the initial value line
            self.scount.vline.set_color('coral')
            self.sgyro_mag.vline.set_color('coral')

            self.scount.on_changed(self.update)
            self.sgyro_mag.on_changed(self.update)

            resetax = plt.axes([0.65, 0.025, 0.1, 0.04])
            self.button_reset = Button(resetax, 'Reset', color=axcolor, hovercolor='0.7')
            self.button_reset.on_clicked(self.reset)

            saveax = plt.axes([0.8, 0.025, 0.1, 0.04])
            self.button_save = Button(saveax, 'Save', color='coral')
            self.button_save.on_clicked(self.save)
            self.button_save.on_clicked(self.change_color)

            plt.show()

        # add data to csv file
        gyro_thresholds.append(
            [
                self.sub,
                self.run,
                round(self.stance_magnitude_thresholds["LF"], 2),
                round(self.stance_magnitude_thresholds["RF"], 2),
                round(self.stance_count_thresholds["LF"], 2),
                round(self.stance_count_thresholds["RF"], 2)
            ]
        )

        with open(os.path.join(self.save_path, 'stance_magnitude_thresholds.csv'), 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(gyro_thresholds)

        self.check_duplicates(os.path.join(self.save_path, 'stance_magnitude_thresholds.csv'))

class AccPlot():
    def __init__(self, read_path, save_path, sub, run):
        self.read_path = read_path
        self.save_path = save_path
        self.sub = sub
        self.run = run
        self.select_vline = None

    def ic_onclick(self, event):
        """ Event handler for click event """
        # global select_vline
        # global selection_x_coordinate
        self.current_ic = event.xdata
        self.vline.set_xdata(self.current_ic)
        self.fig.canvas.draw_idle()

    def save(self, event):
        self.acc_ic[self.foot] = round(self.current_ic, 7)

        plt.savefig(os.path.join(self.save_path,
                                str('imu_ic_' + self.foot + '_.png')),
                    bbox_inches='tight')

    def change_color(self, event):
        self.button_save.color = '0.7'

    def check_duplicates(self, file_path):
        df = pd.read_csv(file_path)
        dup = df.groupby(['run', 'subject']).size() > 1
        if dup.any():
            print('!!!!!!!!!! duplicate entries !!!!!!!!!! see below:')
            print(dup[dup != 0])
        else:
            print("======== No duplicate entries for acc initial contact. ========")

    def acc_ic_plot(self):
        # get the data file
        acc_ic_info = []  # save one subject at a time
        self.acc_ic = {"LF": None, "RF": None}

        for self.foot in ["LF", "RF"]:
            imu_path = os.path.join(self.read_path, self.foot + ".csv")
            self.imu = IMU(imu_path)
            self.imu.gyro_to_rad()
            gyro_mag = np.linalg.norm(self.imu.gyro(), axis=1)
            self.time = self.imu.time()
            accel = np.transpose(self.imu.accel())
            accel_norm = np.linalg.norm(accel, axis=0)

            prominence_threshold = 3

            peaks, _ = sp.signal.find_peaks(accel_norm)
            prominences = sp.signal.peak_prominences(accel_norm, peaks)[0]

            self.auto_detect_ic = self.time[
                peaks[prominences > prominence_threshold][0]
            ]
            self.current_ic = self.auto_detect_ic

            # make plot
            self.fig, self.ax = plt.subplots(figsize=(12, 5))
            self.fig.canvas.mpl_connect("key_press_event", self.ic_onclick)

            plt.plot(self.time, accel[0], c=plt.cm.winter_r(0), label="X")
            plt.plot(self.time, accel[1], c=plt.cm.winter_r(100), label="Y")
            plt.plot(self.time, accel[2], c=plt.cm.winter_r(200), label="Z")
            self.vline = self.ax.axvline(self.current_ic, color='coral', label='Initial Contact')
            plt.title('IMU Initial Contact Detection' +
                        '\n' + self.sub + '  ' + self.run + '  ' + self.foot)
            plt.xlabel("Time (s)")
            plt.ylabel("IMU Acceleration (g)")
            plt.legend()
            self.ax.margins(x=0)

            saveax = plt.axes([0.8, 0.025, 0.1, 0.04])
            self.button_save = Button(saveax, 'Save', color='coral')
            self.button_save.on_clicked(self.save)
            self.button_save.on_clicked(self.change_color)

            plt.show()

        # add data to csv file
        acc_ic_info.append(
            [
                self.sub,
                self.run,
                self.acc_ic["LF"],
                self.acc_ic["RF"],
            ]
        )

        with open(os.path.join(self.save_path, 'imu_initial_contact.csv'), 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(acc_ic_info)

        self.check_duplicates(os.path.join(self.save_path, 'imu_initial_contact.csv'))


if __name__ == "__main__":

    ########## Configurations and initialization

    overwrite = False  # if False: append to existing file 
    with open('path.json') as f:
        paths = json.load(f)
    base_path = os.path.join(paths['data_kiel'], "interim")

    # runs = [
    #     "slow", 
    #     "normal",
    #     "fast"
    # ]

    # subjects = [
    #     "physilog",
    #     "xsens1",
    #     "xsens2"
    # ]

    runs = [
        "treadmill"
    ]

    subjects = [
    "pp010",
    # "pp011",
    # "pp028",
    # "pp071",
    # # "pp077",
    # "pp079",
    # "pp099",
    # "pp101",
    # # "pp105",
    # "pp106",
    # "pp109",
    # # "pp112",
    # # "pp114",
    # "pp122",
    # "pp123",
    # "pp137",
    # # "pp139",
    # "pp145",
    # "pp149",
    # "pp158",
    # "pp165",
    # "pp166"
]

    # if no file, create one. Otherwise append to the existing file
    if not os.path.isfile(os.path.join(base_path, 'stance_magnitude_thresholds.csv')) or overwrite:
        pd.DataFrame(
            columns=[
                "subject",
                "run",
                "stance_magnitude_threshold_left",
                "stance_magnitude_threshold_right",
                "stance_count_threshold_left",
                "stance_count_threshold_right",
            ],
        ).to_csv(
            os.path.join(base_path, "stance_magnitude_thresholds.csv"), index=False
        )

    # if no file, create one. Otherwise append to the existing file
    if not os.path.isfile(os.path.join(base_path, 'imu_initial_contact.csv')) or overwrite:
        pd.DataFrame(
            columns=[
                "subject",
                "run",
                "imu_initial_contact_left",
                "imu_initial_contact_right"
            ],
        ).to_csv(
            os.path.join(base_path, "imu_initial_contact.csv"), index=False
        )

    # gyro_thresholds = []  # if save all runs at once

    for subject_id, subject in enumerate(subjects):
        subject_directory = os.path.join(base_path, subject)
        # only process one run
        runs = ["treadmill"]

        # go through all the runs
        # runs = [
        #     x
        #     for x in os.listdir(subject_directory)
        #     if os.path.isdir(os.path.join(subject_directory, x))
        # ]
        for run_id, run in enumerate(runs):
            print(
                "subject",
                subject_id + 1,
                "/",
                len(subjects),
                "/",
                "run",
                run_id + 1,
                "/",
                len(runs)
            )
            file_directory = os.path.join(subject_directory, run, "imu")

            ##### Gyro stance threshold detection
            # ip = GyroPlot(file_directory, base_path, subject, run)
            # ip.gyro_threshold_slider()

            #### Acc initial contact detection
            ap = AccPlot(file_directory, base_path, subject, run)
            ap.acc_ic_plot()




            # c0 = 8  # initial stance count threshold
            # g0 = 0.7  # initial gyro magnitude threshold
            # # setup
            # delta_c = 1  # stance count resolution
            # delta_g = 0.1  # gyro magnitude resolution

            # # get the data file
            # gyro_thresholds = []  # save one subject at a time
            # stance_magnitude_thresholds = {"LF": None, "RF": None}
            # stance_count_thresholds = {"LF": None, "RF": None}

            # for foot in ["LF", "RF"]:
            #     imu_path = os.path.join(file_directory, foot + ".csv")
            #     imu = IMU(imu_path)
            #     imu.gyro_to_rad()
            #     gyro_mag = np.linalg.norm(imu.gyro(), axis=1)
            #     samples = np.arange(len(gyro_mag))

            #     stance = gyro_threshold_stance(
            #         imu,
            #         stance_magnitude_threshold=g0,
            #         stance_count_threshold=c0,
            #     ).astype(bool)

            #     # make plot
            #     fig, ax = plt.subplots(figsize=(12, 5))
            #     plt.subplots_adjust(left=0.12, bottom=0.3)
            #     l, = plt.plot(samples, gyro_mag, lw=1)
            #     hline = ax.axhline(y=g0, xmin=0.0, xmax=1.0, color='coral', label='Gyro Magnitude Threshold')
            #     stance_shadows = ax.fill_between(samples, 0, 1, where=stance, alpha=0.4,
            #                                      facecolor='skyblue',
            #                                      transform=ax.get_xaxis_transform(), label='Stance')
            #     plt.title('Manual Stance Detection' +
            #               '\n' + runs[run_id] + '  ' + subjects[subject_id] + '  ' + foot)
            #     plt.legend()
            #     plt.ylabel('Gyro Magnitude (rad/s)')
            #     plt.xlabel('Sample Number')
            #     ax.margins(x=0)

            #     axcolor = '0.9'
            #     axcount = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
            #     axgyro_mag = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

            #     scount = Slider(axcount, 'Count Threshold', 3, 15, valinit=c0, valstep=delta_c)
            #     sgyro_mag = Slider(axgyro_mag, 'Gyro Mag. Threshold', 0.1, 2.5, valinit=g0, valstep=delta_g)
            #     # The vline attribute controls the initial value line
            #     scount.vline.set_color('coral')
            #     sgyro_mag.vline.set_color('coral')

            #     scount.on_changed(update)
            #     sgyro_mag.on_changed(update)

            #     resetax = plt.axes([0.65, 0.025, 0.1, 0.04])
            #     button_reset = Button(resetax, 'Reset', color=axcolor, hovercolor='0.7')
            #     button_reset.on_clicked(reset)

            #     saveax = plt.axes([0.8, 0.025, 0.1, 0.04])
            #     button_save = Button(saveax, 'Save', color='coral')
            #     button_save.on_clicked(save)
            #     button_save.on_clicked(change_color)

            #     plt.show()

            # # add data to csv file
            # gyro_thresholds.append(
            #     [
            #         run,
            #         subject,
            #         stance_magnitude_thresholds["LF"],
            #         stance_magnitude_thresholds["RF"],
            #         stance_count_thresholds["LF"],
            #         stance_count_thresholds["RF"],
            #     ]
            # )

            # with open(os.path.join(base_path, 'stance_magnitude_thresholds.csv'), 'a') as f:
            #     writer = csv.writer(f, delimiter=',')
            #     writer.writerows(gyro_thresholds)

            # check_duplicates(os.path.join(base_path, 'stance_magnitude_thresholds.csv'))
