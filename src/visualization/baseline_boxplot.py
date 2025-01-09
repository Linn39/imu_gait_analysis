#### plots acc & gyro magnitude baselines

#### imports ####
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path
import fnmatch
import seaborn as sns
sns.set()

#### functions ####
def save_mag_data(df, save_csv_path, device_name, file_name):
    """
    @param df: dataframe
    @type df: dataframe with acc and gyro xyz columns
    @param save_csv_path: path to save the export file
    @type save_csv_path: folder path
    @param device_name: name of device to be displayed on the plot
    @type device_name: string
    @param file_name: name of export file with acc and gyro mag values
    @type file_name: string
    """
    df['Device'] = device_name

    # calculate acc and gyro magnitudes
    df['Acc_mag'] = np.linalg.norm(df[['AccX', 'AccY', 'AccZ']].values, axis=-1)
    df['Gyr_mag'] = np.linalg.norm(df[['GyrX', 'GyrY', 'GyrZ']].values, axis=-1)

    # save csv files
    df.to_csv(os.path.join(save_csv_path, file_name))


def baseline_boxplot(df, title, save_fig_path):
    """
    @param df: dataframe for plotting
    @type df: dataframe
    @param title: title of plot and file
    @type title: string
    @param save_fig_path: path for saving the figure
    @type save_fig_path: folder path
    """
    sns.set_style("whitegrid", {'axes.edgecolor': 'black'})
    sns.set_context("paper", font_scale=1.8)

    fig1 = plt.figure(figsize=(15, 5))

    if 'Acc' in title:
        h_position = 1
        col_name = 'Acc_mag'
        y_label = 'Accleration Magnitude (g)'

    elif 'Gyr' in title:
        h_position = 0
        col_name = 'Gyr_mag'
        y_label = 'Angular Velocity Magnitude (deg/s)'

    ax = sns.boxplot(x='Device', y=col_name, data=df, #color=plt.cm.winter_r(100),
                fliersize=2)#, #palette='Paired')
    plt.setp(ax.artists, edgecolor='0.4', facecolor='w')
    plt.setp(ax.lines, color='0.4')
    plt.axhline(y=h_position, linewidth=2, color=plt.cm.winter_r(100), alpha=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.ylabel(y_label)
    plt.title(title)

    plt.savefig(os.path.join(save_fig_path, str(title + '.png')), bbox_inches='tight')


def plot_raw_baseline(df, title):
    """
    @param df: dataframe with raw acc and gyro data
    @type df: dataframe
    @param title: title of the plot
    @type title: string
    @return: figure
    @rtype:
    """
    # optional: plot raw data to check if the baseline is flat
    df.plot(figsize=(15, 5),
            color=[plt.cm.winter_r(0), plt.cm.winter_r(100), plt.cm.winter_r(200)])  # cmap=plt.cm.winter_r)
    plt.title(title)


#### main ####
if __name__ == "__main__":

    dataset = 'baselines'
    sensor = 'GaitUp'
    sub = 'baselines14_IMUs'  # just the name of the folder
    data_path = os.path.join(dataset, sensor, sub)

    #find folder for the baseline values in .csv
    with open('../../path.json') as f:
        paths = json.load(f)
    read_folder_path = os.path.join(paths['raw_data'], data_path)

    if not os.path.exists(read_folder_path):
        os.makedirs(read_folder_path)

    # get file names for all baseline raw data and load data
    for file in os.listdir(read_folder_path):
        if fnmatch.fnmatch(file, '*.csv'):
            print(file)
            # device_name = file[:-4]  # remove '.csv' at the end of the file name
            device_name = file[0:3]  # identify the device using the first letters in file name
            df = pd.read_csv(os.path.join(read_folder_path, file))  # read loaded .csv files
            save_mag_data(df, read_folder_path, device_name, device_name + '.csv')

    # read and concat all dfs for plotting
    all_baseline_dfs = [i for i in glob.glob(os.path.join(read_folder_path, '*.csv'))]
    combined_baseline_df = pd.concat([pd.read_csv(f) for f in all_baseline_dfs], sort=False)
    print('num data points:')
    print(len(combined_baseline_df.index))
    print(combined_baseline_df.columns)

    # boxplots
    baseline_boxplot(combined_baseline_df[['Device', 'Acc_mag']],
                     'Acc_mag_' + str(len(combined_baseline_df.index)) + '_samples', read_folder_path)
    baseline_boxplot(combined_baseline_df[['Device', 'Gyr_mag']],
                     'Gyr_mag_' + str(len(combined_baseline_df.index)) + '_samples', read_folder_path)
    plt.show()

    # # optional: plot raw baseline data
    # combined_baseline_df = combined_baseline_df.reset_index()
    # plot_raw_baseline(combined_baseline_df[['AccX', 'AccY', 'AccZ']], 'raw_acc')
    # plot_raw_baseline(combined_baseline_df[['GyrX', 'GyrY', 'GyrZ']], 'raw_gyr')
    # plt.show()
