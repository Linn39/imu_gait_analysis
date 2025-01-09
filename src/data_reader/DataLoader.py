import pandas as pd
import numpy as np
import json
import os.path
import fnmatch
from scipy.io import loadmat


class DataLoader:
    """
    load raw IMU data & save as .csv file
    """

    def __init__(self, data_path, location_kw):
        self.data_path = data_path
        self.location_kw = location_kw
        self.data_df = None

    def load_csv_data(self):
        """load .csv data that are already formatted"""
        # find file containing key word for left or right foot
        no_file = True
        for file in os.listdir(self.data_path):
            if fnmatch.fnmatch(file, self.location_kw + ".csv"):
                file_name = file
                print(file_name)
                no_file = False

        if no_file:
            print("No file for " + self.location_kw + " found.")
            return

        self.data_df = pd.read_csv(os.path.join(self.data_path, file_name))
        return self.data_df

    def load_kiel_data(self):
        """load .mat data from kiel dataset"""
        kw_dict = {"LF": "left_foot", "RF": "right_foot"}
        # find file containing imu data
        no_file = True
        for file in os.listdir(self.data_path):
            if fnmatch.fnmatch(file, "imu*.mat"):
                file_name = file
                print(file_name)
                no_file = False

        mat_data = loadmat(
            os.path.join(self.data_path, file_name), simplify_cells=True
        )[
            "data"
        ]  # dictionary of data

        imu_values = np.concatenate(
            list(map(mat_data.get, ["acc", "gyro"])), 1
        )  # get all imu data
        idx = list(mat_data["imu_location"]).index(
            kw_dict[self.location_kw]
        )  # get index of the location
        self.data_df = pd.DataFrame(
            imu_values[:, :, idx],
            columns=["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"],
        )

        # create artificial timestamp based on sampling rate
        fs = mat_data["fs"]  # get sampling rate
        self.data_df["timestamp"] = np.arange(0, len(self.data_df) / fs, 1 / fs)
        return self.data_df

    def load_GaitUp_data(self):
        """load and reformat data for gait analysis script."""

        # find file containing key word for left or right foot
        no_file = True
        for file in os.listdir(self.data_path):
            if fnmatch.fnmatch(file, "*.csv") and (self.location_kw in file):
                file_name = file
                print(file_name)
                no_file = False

        if no_file:
            print("No file for " + self.location_kw + " found.")
            return

        # extract data columns
        raw_data_df = pd.read_csv(
            os.path.join(self.data_path, file_name), skiprows=5, low_memory=False
        )
        self.data_df = raw_data_df.filter(
            ["Time", "Gyro X", "Gyro Y", "Gyro Z", "Accel X", "Accel Y", "Accel Z"],
            axis=1,
        )
        self.data_df = self.data_df.rename(
            columns={
                "Time": "timestamp",
                "Accel X": "AccX",
                "Accel Y": "AccY",
                "Accel Z": "AccZ",
                "Gyro X": "GyrX",
                "Gyro Y": "GyrY",
                "Gyro Z": "GyrZ",
            }
        )

        self.data_df.drop(
            index=self.data_df.index[0], axis=0, inplace=True
        )  # drop first row with units
        self.data_df.dropna(inplace=True)
        self.data_df = self.data_df.apply(
            pd.to_numeric
        )  # convert all columns of DataFrame to numbers

        return self.data_df

    def load_xsens_data(self):
        # find file containing key word for left or right foot
        no_file = True
        for file in os.listdir(self.data_path):
            if fnmatch.fnmatch(file, "*.csv") and (self.location_kw in file):
                file_name = file
                print(file_name)
                no_file = False

        if no_file:
            print("No file for " + self.location_kw + " found.")
            return

        # extract data columns
        raw_data_df = pd.read_csv(
            os.path.join(self.data_path, file_name), skiprows=7, low_memory=False
        )
        self.data_df = raw_data_df.filter(
            ["SampleTimeFine", "Gyr_X", "Gyr_Y", "Gyr_Z", "Acc_X", "Acc_Y", "Acc_Z"],
            axis=1,
        )
        self.data_df = self.data_df.rename(
            columns={
                "SampleTimeFine": "timestamp",
                "Acc_X": "AccY",
                "Acc_Y": "AccX",
                "Acc_Z": "AccZ",
                "Gyr_X": "GyrY",
                "Gyr_Y": "GyrX",
                "Gyr_Z": "GyrZ",
            }
        )

        self.data_df = self.data_df.apply(
            pd.to_numeric, errors="coerce"
        )  # convert all columns of DataFrame to numbers
        self.data_df.dropna(
            inplace=True
        )  # in case there are non-numeric values being converted to NaN

        try:  # correct the timestamps in case the time counting restarts
            # find the index where the timestamp restarts
            restart_index = self.data_df[
                self.data_df["timestamp"] < self.data_df["timestamp"].shift(1)
            ].index[0]

            # calculate the offset to add to subsequent timestamps
            offset = 4294967295 + 1  # the timestamp always wraps at 2^32 - 1

            # add the offset to subsequent values
            self.data_df.loc[restart_index:, "timestamp"] += offset

            # verify that the timestamp column contains only incremental values
            assert np.all(
                np.diff(self.data_df["timestamp"]) > 0
            ), "Timestamps are not in strictly increasing order"

        except IndexError:
            # pass if all timestamps are already correct
            pass

        self.data_df["timestamp"] = (
            self.data_df["timestamp"] * 1e-6
        )  # convert time from microsecond to second
        self.data_df["GyrX"] = self.data_df["GyrX"] * (
            -1
        )  # invert gyro Y axis for gait event detection
        self.data_df["AccX"] = self.data_df["AccX"] / 9.8 * (-1)
        self.data_df["AccY"] = self.data_df["AccY"] / 9.8
        self.data_df["AccZ"] = self.data_df["AccZ"] / 9.8
        return self.data_df

    def load_EXLs3_data(self):
        folder_path = self.data_path

        print("folder_path:     ", folder_path)
        # find file containing key word for left or right foot
        if self.location_kw == "LF":
            for file in os.listdir(folder_path):
                if fnmatch.fnmatch(file, "Gait - L*"):
                    file_name = file
        elif self.location_kw == "RF":
            for file in os.listdir(folder_path):
                if fnmatch.fnmatch(file, "Gait - R*"):
                    file_name = file

        self.data_df = pd.read_fwf(os.path.join(folder_path, file_name), header=3)

        # replace column names
        self.data_df.columns = [
            "Header",
            "AccX",
            "AccY",
            "AccZ",
            "GyrX",
            "GyrY",
            "GyrZ",
            "Counter",
            "Checksum",
            "Nr",
            "Filteredtss",
            "Unfilteredtss",
            "timestamp",
            "FreqHz",
            "m_xG",
            "m_yG",
            "m_zG",
            "q0",
            "q1",
            "q2",
            "q3",
        ]

        return self.data_df

    def load_bonsai_data(self):
        foot_name = self.location_kw
        folder_path = self.data_path

        # load Bonsai data with key word foot_name
        # find file containing key word for left or right foot
        no_file = True
        if foot_name == "LF":
            for file in os.listdir(folder_path):
                if fnmatch.fnmatch(file, "*I-L9H*") or fnmatch.fnmatch(file, "*I-2VZ*"):
                    file_name = file
                    print(file_name)
                    no_file = False
        elif foot_name == "RF":
            for file in os.listdir(folder_path):
                if fnmatch.fnmatch(file, "*I-0GN*"):
                    file_name = file
                    print(file_name)
                    no_file = False

        if no_file:
            print("No file for " + foot_name + " found.")

        # extract data columns
        raw_data_df = pd.read_csv(os.path.join(folder_path, file_name))
        # adjust the xyz coordinates to match the script for EXLs3
        data_df = raw_data_df.filter(
            ["timestamp", "accY", "accX", "accZ", "gyrY", "gyrX", "gyrZ"], axis=1
        )
        data_df = data_df.rename(
            columns={
                "accY": "AccX",
                "accX": "AccY",
                "accZ": "AccZ",
                "gyrY": "GyrX",
                "gyrX": "GyrY",
                "gyrZ": "GyrZ",
            }
        )

        # convert the gyro unit from rad/s to deg/s
        data_df["GyrX"] = data_df["GyrX"] * 180 / np.pi
        data_df["GyrY"] = data_df["GyrY"] * 180 / np.pi
        data_df["GyrZ"] = data_df["GyrZ"] * 180 / np.pi

        # scale down *10 to match the EXLs3 data
        data_df["GyrX"] = data_df["GyrX"] / 10
        data_df["GyrY"] = data_df["GyrY"] / 10 * (-1)
        data_df["GyrZ"] = data_df["GyrZ"] / 10

        data_df["AccX"] = data_df["AccX"] / 9.8
        data_df["AccY"] = data_df["AccY"] / 9.8 * (-1)
        data_df["AccZ"] = data_df["AccZ"] / 9.8

        self.data_df = data_df

        return self.data_df

    def load_MM_data(self):
        read_path = self.data_path

        no_file_acc = True
        no_file_gyro = True
        for file in os.listdir(read_path):
            if fnmatch.fnmatch(file, self.location_kw + "*Accelerometer.csv"):
                file_name_acc = file
                print(file_name_acc)
                no_file_acc = False
            if fnmatch.fnmatch(file, self.location_kw + "*Gyroscope.csv"):
                file_name_gyro = file
                print(file_name_gyro)
                no_file_gyro = False

        if no_file_acc:
            print("No acc file for " + self.location_kw + " found.")
        if no_file_gyro:
            print("No gyro file for " + self.location_kw + " found.")
            return

        # extract data columns
        acc_data_df = pd.read_csv(os.path.join(read_path, file_name_acc))
        gyro_data_df = pd.read_csv(os.path.join(read_path, file_name_gyro))
        acc_data_df = acc_data_df.filter(
            ["epoc (ms)", "elapsed (s)", "x-axis (g)", "y-axis (g)", "z-axis (g)"],
            axis=1,
        )
        gyro_data_df = gyro_data_df.filter(
            ["x-axis (deg/s)", "y-axis (deg/s)", "z-axis (deg/s)"], axis=1
        )
        raw_data_df = pd.concat([acc_data_df, gyro_data_df], axis=1)
        raw_data_df = raw_data_df.dropna()  # match the number of data points
        raw_data_df["epoc (ms)"] = (
            raw_data_df["epoc (ms)"] / 1000
        )  # convert ms to s for the unix timestamp

        # adjust the xyz coordinates to match the script for EXLs3
        self.data_df = raw_data_df.rename(
            columns={
                "epoc (ms)": "unix_timestamp",
                "elapsed (s)": "timestamp",
                "x-axis (g)": "AccX",
                "y-axis (g)": "AccY",
                "z-axis (g)": "AccZ",
                "x-axis (deg/s)": "GyrX",
                "y-axis (deg/s)": "GyrY",
                "z-axis (deg/s)": "GyrZ",
            }
        )

        return self.data_df

    def load_movi_data(self):
        folder_path = self.data_path
        foot_name = self.location_kw

        # read acc and gyro data from .csv files
        acc_df = pd.read_csv(
            os.path.join(folder_path, foot_name, "acc.csv"),
            header=None,
            sep=";",
            names=["AccX", "AccY", "AccZ"],
        )
        gyro_df = pd.read_csv(
            os.path.join(folder_path, foot_name, "angularrate.csv"),
            header=None,
            sep=";",
            names=["GyrX", "GyrY", "GyrZ"],
        )

        # concat acc and gyro files
        data_df = pd.concat([acc_df, gyro_df], axis=1)

        # add timestamps
        data_df["timestamp"] = np.arange(0, len(data_df) / 256, 1 / 256)
        data_df = data_df.iloc[::2]  # even

        # scale down *10 to match the EXLs3 data
        data_df["GyrX"] = data_df["GyrX"] * 0.07000000066757203  # / 16.384
        data_df["GyrY"] = data_df["GyrY"] * 0.07000000066757203  # / 16.384
        data_df["GyrZ"] = data_df["GyrZ"] * 0.07000000066757203  # / 16.384

        data_df["AccX"] = data_df["AccX"] / 2048
        data_df["AccY"] = data_df["AccY"] / 2048
        data_df["AccZ"] = data_df["AccZ"] / 2048

        self.data_df = data_df

        return self.data_df

    def load_portabiles_data(self):
        folder_path = self.data_path
        foot_name = self.location_kw

        # load Portabiles data with key word foot_name
        # find file containing key word for left or right foot
        no_file = True
        if foot_name == "LF":
            for file in os.listdir(folder_path):
                if fnmatch.fnmatch(file, "*92B9*.csv"):
                    file_name = file
                    no_file = False
        elif foot_name == "RF":
            for file in os.listdir(folder_path):
                if fnmatch.fnmatch(file, "*4061*.csv"):
                    file_name = file
                    no_file = False

        if no_file:
            print("No file for " + foot_name + " found.")

        # extract data columns
        raw_data_df = pd.read_csv(os.path.join(folder_path, file_name), skiprows=1)
        # adjust the xyz coordinates to match the script for EXLs3
        data_df = raw_data_df.filter(
            ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
            axis=1,
        )
        data_df = data_df.rename(
            columns={
                "acc_x": "AccX",
                "acc_y": "AccY",
                "acc_z": "AccZ",
                "gyro_x": "GyrX",
                "gyro_y": "GyrY",
                "gyro_z": "GyrZ",
            }
        )

        # # convert the gyro unit from rad/s to deg/s
        # data_df['GyrX'] = data_df['GyrX'] * 180 / np.pi
        # data_df['GyrY'] = data_df['GyrY'] * 180 / np.pi
        # data_df['GyrZ'] = data_df['GyrZ'] * 180 / np.pi
        #
        # scale down *10 to match the EXLs3 data
        data_df["GyrX"] = data_df["GyrX"] / 16.384
        data_df["GyrY"] = data_df["GyrY"] / 16.384
        data_df["GyrZ"] = data_df["GyrZ"] / 16.384

        data_df["AccX"] = data_df["AccX"] / 2048
        data_df["AccY"] = data_df["AccY"] / 2048
        data_df["AccZ"] = data_df["AccZ"] / 2048

        self.data_df = data_df

        return self.data_df

    def load_shimmer_data(self):
        folder_path = self.data_path
        foot_name = self.location_kw

        # load data with key word foot_name
        # find file containing key word for left or right foot
        no_file = True
        if foot_name == "LF":
            for file in os.listdir(folder_path):
                if fnmatch.fnmatch(file, "*LF*"):
                    file_name = file
                    no_file = False
        elif foot_name == "RF":
            for file in os.listdir(folder_path):
                if fnmatch.fnmatch(file, "*RF*"):
                    file_name = file
                    no_file = False

        if no_file:
            print("No file for " + foot_name + " found.")

        # extract data columns
        raw_data_df = pd.read_csv(
            os.path.join(folder_path, file_name), skiprows=0, header=1
        )

        # adjust the xyz coordinates to match the analysis script
        data_df = raw_data_df.filter(
            [
                foot_name + "_Timestamp_Unix_CAL",
                foot_name + "_Accel_WR_X_CAL",
                foot_name + "_Accel_WR_Y_CAL",
                foot_name + "_Accel_WR_Z_CAL",
                foot_name + "_Gyro_X_CAL",
                foot_name + "_Gyro_Y_CAL",
                foot_name + "_Gyro_Z_CAL",
            ],
            axis=1,
        )

        data_df = data_df.rename(
            columns={
                foot_name + "_Timestamp_Unix_CAL": "timestamp",
                foot_name + "_Accel_WR_X_CAL": "AccX",
                foot_name + "_Accel_WR_Y_CAL": "AccY",
                foot_name + "_Accel_WR_Z_CAL": "AccZ",
                foot_name + "_Gyro_X_CAL": "GyrX",
                foot_name + "_Gyro_Y_CAL": "GyrY",
                foot_name + "_Gyro_Z_CAL": "GyrZ",
            }
        )

        data_df.drop(0, inplace=True)
        data_df.reset_index(drop=True, inplace=True)
        data_df = data_df.apply(
            pd.to_numeric
        )  # convert all columns of DataFrame to numbers

        # change unix timestamps from ms to s
        data_df.loc[:, "timestamp"] = data_df["timestamp"] / 1000

        # convert acc m/s2 to g with gravity in Berlin
        data_df.loc[:, "AccX"] = data_df["AccX"] / 9.813
        data_df.loc[:, "AccY"] = data_df["AccY"] / 9.813
        data_df.loc[:, "AccZ"] = data_df["AccZ"] / 9.813

        self.data_df = data_df

        return self.data_df

    def cut_data(self, start_cut, end_cut, by_timestamp=False):
        try:
            if by_timestamp:
                # use the timestamp column as reference to cut the data
                # e.g., for the Xsens DOTs, the timestamp is synchronized, but index and sample number could be different
                self.data_df = self.data_df[
                    (self.data_df["timestamp"] >= start_cut)
                    & (self.data_df["timestamp"] < end_cut)
                ]
            else:
                # use the index as reference to cut the data
                self.data_df = self.data_df[
                    (self.data_df.index >= start_cut) & (self.data_df.index < end_cut)
                ]
            # if dataframe is not empty
            if not self.data_df.empty:
                return self.data_df
            else:
                print("Empty dataframe, check if cutting positions are correct.")
                # exit the program
                exit()

        except AttributeError:
            print("Could not cut: Data not loaded yet.")
        else:
            print("Data successfully cut.")

    def save_data(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        try:
            self.data_df.to_csv(save_path + "/" + self.location_kw + ".csv")
        except AttributeError:
            print("Could not save to csv: Data not loaded yet.")
        else:
            print("IMU data loaded and saved.")

    # def get_data(self):
    #     return self.data_df


if __name__ == "__main__":
    pass
