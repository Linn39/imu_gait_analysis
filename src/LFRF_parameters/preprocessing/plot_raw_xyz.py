#### imports ####
import sys, os
sys.path.append("./src")
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import json
import seaborn as sns
sns.set()

from data_reader.DataLoader import DataLoader


# print(os.getcwd())

#### functions ####
def get_acc_gyr(df, sensor):
    """
    extract acc and gyr columns and add sensor name to the column headers
    :param df: the raw data
    :param sensor: sensor name to be added to the column headers
    :return: extracted dataframe
    """
    df = df[['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']]  # get acc and gyr columns
    df = df.add_suffix('_' + sensor)
    return df


def plot_acc_gyr(df, columns, title, save_fig_path):
    """
    plot raw sensor data from xyz axis
    :param df: dataframe contatining data, column names are used as legends
    :param columns: selsect columns to be plotted. e.g. ['AccX', 'AccY', 'AccZ']
    :param title: title of the figure
    :param save_fig_path: folder path for saving the figure
    :return: saves the figure in .png
    """

    sns.set_style("whitegrid", {'axes.edgecolor': 'black'})
    sns.set_context("paper", font_scale=1.8)

    plot_df = df[columns].copy()
    if 'timestamp' in plot_df.columns:
        plot_df.set_index('timestamp', drop=True, inplace=True)  # use timestamp as x axis
        plot_df.plot(figsize=(15, 5))
            # color=[plt.cm.winter_r(0), plt.cm.winter_r(100), plt.cm.winter_r(200)])  # cmap=plt.cm.winter_r)
        plt.xlabel('Time (s)')
    # plt.ylim(-10, 10)
    else:
        plot_df.plot(figsize=(15, 5))
            # color=[plt.cm.winter_r(0), plt.cm.winter_r(100), plt.cm.winter_r(200)])  # cmap=plt.cm.winter_r)
        plt.xlabel('Samples')

    if 'Acc' in title:
        plt.ylabel('Acceleration (g)')
        acc_mag = np.linalg.norm(plot_df.values, axis=-1)
        plt.title(title + '\n Acc_mag = ' +
                  '{:.2f}'.format(round(np.mean(acc_mag), 2)) + '    '
                                                                'num. samples = ' + str(len(plot_df.index)))
        # print('acc mag: ' + str(np.mean(acc_mag)))
        print('num. acc samples = ' + str(len(plot_df.index)))
    elif 'Gyr' in title:
        plt.ylabel('Angular Velocity (Degrees/s)')
        gyro_mag = np.linalg.norm(plot_df.values, axis=-1)
        plt.title(title + '\n Gyro_mag = ' +
                  '{:.2f}'.format(round(np.mean(gyro_mag), 2)) + '    '
                                                                 'num. samples = ' + str(len(plot_df.index)))
        # print('gyro mag: ' + str(np.mean(gyro_mag)))
        print('num. gyro samples = ' + str(len(plot_df.index)) + '\n')
    else:
        plt.ylabel('Data')

    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    plt.savefig(os.path.join(save_fig_path, str(title + '.png')), bbox_inches='tight')
    # plt.show()


#### main ####
if __name__ == "__main__":
    save_temp = False  # True
    from_interim = False  # load interim data
    dataset = "data_kiel_val"  # "data_TRIPOD"
    exp_condition = "gait1"  # 'PWS+20'
    sub_list = ['pp001']  # no for loop: all raw data should be inspected manually
    # sub_list = [
    # "pp010",
    # "pp011",
    # "pp028",
    # "pp071",
    # "pp079",
    # "pp099",
    # "pp105", 
    # "pp106",
    # # "pp114",  # has no timestamp markers 
    # "pp137",
    # "pp139", 
    # "pp158",
    # "pp165",
    # # "pp166",  # only two timestamp markers
    # "pp077",
    # "pp101",
    # "pp109",
    # "pp112", 
    # "pp122",
    # "pp123",
    # "pp145",
    # "pp149"
    # ]

    # sub_list = [
    #     "Sub_AL",
    #     "Sub_BK",
    #     "Sub_CP",
    #     "Sub_DF",
    #     "Sub_EN",
    #     "Sub_FZ",
    #     "Sub_GK",
    #     "Sub_HA",
    #     "Sub_KP",
    #     "Sub_LU",
    #     "Sub_OD",
    #     "Sub_PB",
    #     "Sub_RW",
    #     "Sub_SN",
    #     "Sub_YU"
    # ]

    for sub in sub_list:
        read_data_path = os.path.join(sub, exp_condition, "imu")  # folder containing the raw data
        save_data_path = read_data_path  # folder to export the loaded data. Usually follows the read_data_path pattern.

        # define path names
        with open('path.json') as f:
            paths = json.load(f)
        read_folder_path = os.path.join(paths[dataset], "raw", read_data_path)
        print("Read path:", read_folder_path)

        #### for treadmill: load timestamps for cutting the data ####
        if exp_condition == "treadmill":
            speed_timestamps_df = pd.read_csv(os.path.join(paths[dataset], "raw", "treadmill_timestamps.csv"))
            start, end = speed_timestamps_df[speed_timestamps_df["sub"] == sub].filter(["start2", "end2"]).values[0] * 200    # 200 Hz
            save_data_path =  os.path.join(sub, f"{exp_condition}_speed2", "imu")   # save segment to a folder with speed name
        ########

        save_folder_path = os.path.join(paths[dataset], "interim", save_data_path)
        print("Save path:", save_folder_path)

        # select IMU locations to load
        IMU_loc_list = ['LF', 'RF']
        for loc in IMU_loc_list:
            if from_interim:  # load interim data
                df_loc = pd.read_csv(os.path.join(read_folder_path, loc + ".csv"))
            else:  # load from /raw (& save file to the interim folder)
                data_loader = DataLoader(read_folder_path, loc)
                # df_loc = data_loader.load_GaitUp_data()   # load gaitup data
                df_loc = data_loader.load_kiel_data()   # load kiel IMU data
                # df_loc = data_loader.cut_data(start, end)  # (if necessary: segment data)
                data_loader.save_data(save_folder_path)  # save re-formatted data into /interim folder

            # if the IMU data is loaded, plot the signals
            try:
                plot_acc_gyr(df_loc, ['AccX', 'AccY', 'AccZ'], f'raw_Acc_{loc}', save_folder_path)
                plot_acc_gyr(df_loc, ['GyrX', 'GyrY', 'GyrZ'], f'raw_Gyr_{loc}', save_folder_path)
            except TypeError:
                print("No data to plot.")
        plt.show()
