import fnmatch
import os
import pandas as pd
from LFRF_parameters.pipeline.abstract_pipeline_components import AbstractDataLoader
from visualization.plot import *
import matplotlib.pyplot as plt

class PhysilogDataLoader(AbstractDataLoader):
    """
    Loads raw Physilog data, extracts relevant columns / data, the export should be saved in /data/interim/
    """

    def __init__(self, data_path, location_kw):
        super().__init__(data_path, location_kw)
        #first_ic = self.get_ic()

    # def load(self, csv_folder):
    #     for file_name in os.listdir(csv_folder):
    #         if fnmatch.fnmatch(file_name, '*.csv'):
    #             csv_file = os.path.join(csv_folder, file_name)
    #             self.data[file_name.split(".")[0]] = IMU(csv_file)

    def load(self):
        """load and reformat data for gait analysis script.
        """

        # find file containing key word for left or right foot
        no_file = True
        for file in os.listdir(self.data_path):
            if fnmatch.fnmatch(file, '*' + self.location_kw + '*.csv'):
                file_name = file
                print(file_name)
                file_path = os.path.join(self.data_path, file_name)
                no_file = False

        if no_file:
            print('No file for ' + self.location_kw + ' found.')
            return

        # extract data columns
        raw_data_df = pd.read_csv(os.path.join(self.data_path, file_name), skiprows=5, low_memory=False)
        self.data_df = raw_data_df.filter(['Time', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Accel X', 'Accel Y', 'Accel Z'], axis=1)
        self.data_df = self.data_df.rename(columns={'Time': 'timestamp',
                                'Accel X': 'AccX',
                                'Accel Y': 'AccY',
                                'Accel Z': 'AccZ',
                                'Gyro X': 'GyrX',
                                'Gyro Y': 'GyrY',
                                'Gyro Z': 'GyrZ'})

        self.data_df = self.data_df.drop(0)
        self.data_df = self.data_df.apply(pd.to_numeric)  # convert all columns of DataFrame to numbers



    def cut_data(self, start_cut, end_cut):
        try:
            self.data_df = self.data_df[(self.data_df.index >= start_cut) & (self.data_df.index < end_cut)]
        except AttributeError:
            print('Could not cut: Data not loaded yet.')
        else:
            print("Data successfully cut.")

    def save_data(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        try:
            save_file_path = save_path + '/' + self.location_kw + '.csv'
            self.data_df.to_csv(save_file_path)
            self.interim_path = save_file_path
        except AttributeError:
            print("Could not save to csv: Data not loaded yet.")
        else:
            print('GaitUp data loaded and saved.')

    def get_ic(self):
        # search region for IC is on the right foot within the first 20 seconds
        plot_accel_gyro(self.data_df["RF"])
        show()
        plt.plot(np.linalg.norm(self.data_df["RF"].accel()[0:20*128], axis=1))
        plt.plot(np.gradient(np.linalg.norm(self.data_df["RF"].accel()[0:20*128], axis=1)))
        plt.show()
        exit()
        return self.data_df["RF"].time(np.argmax(np.max(self.data_df["RF"].accel()[0:20 * 128], axis=1)))

    def get_interim_path(self):
        return self.interim_path
