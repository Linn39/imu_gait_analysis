import numpy as np
import pandas as pd
import copy


class IMU:
    """
        Takes a CSV file exported from the raw Physilog data
        or from a MATLAB dump 
        or from a json and loads its content
        @param file_type either "Physilog" or "MATLAB"
        @param from_json if True directly read datafram from json. This is used to serialize and deserialize an IMU object
    """
    def __init__(self, data_source, file_type="imu", from_json=False):
        if from_json:
            self.data = pd.read_json(data_source)
        elif (file_type == "imu"):
            print(data_source)
            self.data = pd.read_csv(data_source)
            self.data['timestamp'] = self.data['timestamp'] - self.data.loc[self.data.index[0], 'timestamp']
            #self.data.drop(columns=list(self.data.head())[-1], inplace=True)
        elif (file_type == "MATLAB"):
            self.data = pd.read_csv(data_source)
            self.data.drop(columns=list(self.data.head())[1], inplace=True)
            self.data.columns = ["timestamp", "GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]

    """
        Getter methods for timeseries of multidimensional data
    """
    def time(self, i=None):
        if i is None:
            return np.array(self.data["timestamp"])

        return np.array(self.data["timestamp"])[i]

    def accel(self, i=None):
        if i is None:
            return np.array(self.data[["AccX", "AccY", "AccZ"]])

        return np.array(self.data[["AccX", "AccY", "AccZ"]])[i]

    def accel_mag(self):
        return np.linalg.norm(self.data[['AccX', 'AccY', 'AccZ']].values, axis=-1)

    def accel_df(self):
        df = pd.DataFrame(self.accel(), columns=["AccX", "AccY", "AccZ"], index=self.time())
        df.index = pd.to_datetime((df.index.array * 1e9).astype("int64"))
        df.index.name = "time"
        return df

    def gyro(self, i=None):
        if i is None:
            return np.array(self.data[["GyrX", "GyrY", "GyrZ"]])

        return np.array(self.data[["GyrX", "GyrY", "GyrZ"]])[i]

    def gyro_mag(self):
        return np.linalg.norm(self.data[['GyrX', 'GyrY', 'GyrZ']].values, axis=-1)

    def quat(self, i=None):
        if i is None:
            return np.array(self.data[["Quat W", "Quat X", "Quat Y", "Quat Z"]])

        return np.array(self.data[["Quat W", "Quat X", "Quat Y", "Quat Z"]])[i]

    def event(self, i=None):
        if i is None:
            return np.array(self.data["Event"])

        return np.array(self.data["Event"])[i]

    def press(self, i=None):
        if i is None:
            return np.array(self.data["Pressure"])

        return np.array(self.data["Pressure"])[i]

    def temp(self, i=None):
        if i is None:
            return np.array(self.data["Temperature"])

        return np.array(self.data["Temperature"])[i]

    """
        Calculate the sampling rate mean and standard deviation
    """
    def check_sampling(self):
        T = (np.append(self.time(), [0]) - np.append([0], self.time()))[1:-1]
        f = 1 / T
        mean = np.mean(f)
        stddev = np.std(f)
        # hist = plt.hist(f, bins='auto')
        # plt.show()
        return mean, stddev

    """
        Calculate acceleration variance
    """
    def accel_variance(self):
        return np.var(self.accel, axis=0)

    """
        Zero-bases timestamps
    """
    def zero_base_time(self):
        self.data["timestamp"] -= self.data["timestamp"][0]

    """
        Shift timestamps -- used for synchronization
    """
    def time_shift(self, time_offset):
        self.data["timestamp"] += time_offset

    """
        Crops data to timeframe between min_time and max_time. Attention: all other data is dropped
    """
    def crop(self, start_cut, end_cut, inplace=True):
        if inplace:
            self.data = self.data[(self.data.index >= start_cut) & (self.data.index < end_cut)]
            self.data.reset_index(drop=True, inplace=True)
        else:
            self_copy = copy.deepcopy(self)
            self_copy.data = self_copy.data[(self_copy.data.index >= start_cut) & (self_copy.data.index < end_cut)]
            self_copy.data.reset_index(drop=True, inplace=True)
            return self_copy

    """
        Crops data using timestamps. Attention: all other data is dropped
    """

    def crop_by_time(self, start_cut, end_cut, inplace=True):
        if inplace:
            self.data = self.data[(self.data.timestamp >= start_cut) & (self.data.timestamp < end_cut)]
            self.data.reset_index(drop=True, inplace=True)
        else:
            self_copy = copy.deepcopy(self)
            self_copy.data = self_copy.data[(self_copy.data.timestamp >= start_cut) & (self_copy.data.timestamp < end_cut)]
            self_copy.data.reset_index(drop=True, inplace=True)
            return self_copy


    """
        Performs a bias calibration on the given timeframe for gyroscope data.
        It is assumed, that the IMU is not rotating during this period
    """
    def calibrate(self, min_time, max_time):
        calibration_data = self.data[(self.data["timestamp"] >= min_time) & (self.data["timestamp"] < max_time)]
        gyro_mean = np.mean(calibration_data[["GyrX", "GyrY", "GyrZ"]])
        self.data[["GyrX", "GyrY", "GyrZ"]] -= gyro_mean

    """
        Exports timestamps, gyroscope and acceleration data to a CSV file
    """
    def export_table(self, path):
        self.data.to_csv(path_or_buf=path,
                         columns=["timestamp", "GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"],
                         header=["timestamp", "GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"],
                         index=False)

    """
        Converts gyroscope measurements from degree to radiants
    """
    def gyro_to_rad(self):
        self.data[["GyrX", "GyrY", "GyrZ"]] *= np.pi / 180

    """
        Converts gyroscope measurements from radiants to degrees
    """
    def gyro_to_degree(self):
        self.data[["GyrX", "GyrY", "GyrZ"]] /= np.pi / 180

    """
        Converts acceleration measurements form g to meters per square seconds
    """
    def acc_to_meter_per_square_sec(self):
        self.data[["AccX", "AccY", "AccZ"]] *= 9.80665

    """
        Converts acceleration measurements meters per square seconds to g
    """
    def acc_to_g(self):
        self.data[["AccX", "AccY", "AccZ"]] /= 9.80665

    """
        Resamples the dataframe to a given frequency (returns a copy)
    """
    def resample(self, f):
        self_copy = copy.deepcopy(self)
        self_copy.data = self_copy.data.set_index("timestamp")
        self_copy.data.index = pd.to_datetime((self_copy.data.index.array * 1e9).astype("int64"))
        self_copy.data.index.name = "timestamp"
        self_copy.data = self_copy.data.resample(pd.Timedelta(1.0 / f, unit="s")).mean().interpolate()
        self_copy.data.index = (self_copy.data.index - pd.to_datetime(0)).total_seconds()
        self_copy.data.reset_index(inplace=True)
        return self_copy

    """
        Takes the longest possible period with gradient smaller than max_gradient and calculates bias from this period.
        Subtracts this bias from the whole recording
    """
    def dedrift_gyro(self, max_gradient=0.02):
        grad = np.abs(np.gradient(self.gyro(), axis=0))

        below_threshold = np.all(grad <= max_gradient, axis=1)

        runs = [(0, 1)]
        for i, e in list(enumerate(below_threshold))[1:]:
            if e and below_threshold[i-1]:
                runs[-1] = (runs[-1][0], runs[-1][1]+1)
            elif e:
                runs.append((i, 1))

        runs = sorted(runs, key=lambda x: x[1], reverse=True)

        longest_run = runs[0]

        run_idx = np.arange(0, longest_run[1]) + longest_run[0]

        drift = np.mean(self.gyro()[run_idx], axis=0)

        self.data[["GyrX", "GyrY", "GyrZ"]] -= drift

    def get_data(self):
        return self.data
