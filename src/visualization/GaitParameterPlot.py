import os
import re
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

sns.set(style="whitegrid")
sns.set_palette(["#377eb8", "#d95f02", "#4daf4a"])

from features.FeatureBuilder import FeatureBuilder


class GaitParameterPlot:
    def __init__(
        self,
        data_base_path,
        sub_list,
        run_list,
        run_name,
        drop_turning_interval,  # boolean
    ) -> None:
        """init

        Args:
            data_base_path (str): path to data
            sub_list (list): list of subjects
            run_list (list): list of runs
            run_name (str): column name of runs in the dataset
            drop_turning_interval (Boolean): whether to drop the turning intervals
        """
        self.data_base_path = data_base_path
        self.features_base_path = os.path.join(
            self.data_base_path, "processed", "features_no_abs_SI"
        )
        self.sub_list = sub_list
        self.run_list = run_list
        self.run_name = run_name
        self.drop_turning_interval = drop_turning_interval

        # load data for the plots
        self.window_sz = 10
        self.window_slide = 2
        self.feature_builder = FeatureBuilder(
            data_base_path,
            self.sub_list,
            self.run_list,
            self.run_name,
            drop_turning_interval,
        )
        self.load_all_strides()  # stride-by-stride gait parameters across entire session
        self.load_across_session_features()  # aggregated gait parameters across entire session
        # self.load_windowed_features()  # aggregated gait parameters by windows
        # self.load_windowed_strides()  # stride-by-stride gait parameters, labeld with windows

    def load_windowed_features(self):
        """Load windowed gait parameters features
        First build then load to make sure the setup is consistent
        """
        self.feature_builder.build_features(
            self.data_base_path,
            self.window_sz,
            self.window_slide,
            aggregate_windows=True,
            save_unwindowed_df=True,
        )

        # load windowed df
        self.windowed_features = pd.read_csv(
            os.path.join(
                self.features_base_path,
                f"agg_windows_{self.window_sz}_{self.window_slide}.csv",
            )
        )

    def load_across_session_features(self):
        """Load across session gait parameter features
        First build then load to make sure the setup is consistent
        """
        # collect aggregated features of all sub_list and visits
        self.feature_builder.collect_across_sessions()

        # get data for the chosen subject
        self.across_session_features = pd.read_csv(
            os.path.join(
                self.features_base_path,
                f"across_sessions_all.csv",
            )
        )
        self.across_session_features_LR = {}
        for foot in ["left", "right"]:
            self.across_session_features_LR[foot] = pd.read_csv(
                os.path.join(
                    self.features_base_path,
                    f"across_sessions_{foot}.csv",
                )
            )

    def load_all_strides(self):
        """Load all stride-by-stride data
        First build then load to make sure the setup is consistent
        """
        self.feature_builder.collect_all_strides()
        self.all_strides_df = pd.read_csv(
            os.path.join(self.features_base_path, "df_all.csv")
        )

    def load_windowed_strides(self):
        """Load all stride-by-stride data organized in windows
        First build then load to make sure the setup is consistent
        """
        self.feature_builder.collect_all_windows(self.window_sz, self.window_slide)
        self.windowed_all_strides_df = pd.read_csv(
            os.path.join(self.features_base_path, "windows_df_all.csv")
        )

    def load_all_ref_subs(self):
        """Load gait features from reference control subjects
            from the Kiel dataset
            for the radar plot

        Returns:
            dataframe: dataframe containing gait features from control subjects
        """
        # concatenate data from all subjects and use subjecct alias "all_controls"
        ref_data = pd.read_csv(
            os.path.join(
                self.data_base_path,
                "..",
                "data_kiel",
                "processed",
                "features_no_abs_SI",
                "across_sessions_all.csv",
            )
        )
        ref_data["sub"] = "all_controls"
        ref_data = ref_data[ref_data["treadmill_speed"] == "treadmill_speed2"].copy()
        # ref_data.drop("treadmill_speed", axis=1, inplace=True)

        return ref_data.select_dtypes(include=["int", "float"]).mean(axis=0)

    def count_num_per_run(self, df):
        """Get number of strides or windows (depending on the input dataframe) for each run

        Args:
            df (DataFrame): dataframe that contains strides or windows labeled with runs

        Returns:
            str: string that summarizes count for each run
        """

        # count for each run
        count_df = df.groupby(self.run_name)[self.run_name].count()

        # create summary string for the counts
        run_count_str = ""
        for run in count_df.index:
            # run_count_str += f"{run} = {count_df.loc[run, self.run_name].values[0]}, "
            run_count_str += f"{run} = {count_df[run]}, "

        return run_count_str[:-2]

    def plot_windowed_feature_distribution(self, sub):
        """Plot distribution of all windowed features for the subject of interest

        Args:
            sub (str): subject
        """
        # get data for the chosen subject
        windows_df = self.windowed_features[self.windowed_features["sub"] == sub].copy()

        # gorup by visit
        numeric_df = windows_df.drop(columns=["sub", "cadence_avg"]).copy()
        plot_df = numeric_df.groupby(self.run_name)
        # Create a box plot of the DataFrame
        window_boxplot = plot_df.boxplot(rot=90, figsize=(15, 10))

        plt.suptitle(
            f"Distribution of values in windowed gait parameters \n \
            {sub}, n windows: {self.count_num_per_run(numeric_df)}"
        )

        plt.show()

    def reverse_string(self, s):  # convert the string in reverse order
        return s[::-1]

    def radar_plot_px(self, sub, by_window, save_fig):
        """Make radar plot for the chosen subject visits 1 and 2 with plotly express (px)

        Args:
            sub (str): subject
            by_window (Boolean): whether to use windowed features or
                aggregated features across the entire walking session
            save_fig (Boolean): whether to save the figure

        Returns:
            _type_: _description_
        """
        if by_window:  # aggregate by windows
            title_suffix = "by_window"

            sub_windowed_df = self.windowed_features[
                self.windowed_features["sub"] == sub
            ].copy()
            mean_windowed_df = (
                sub_windowed_df.groupby(self.run_name).mean().reset_index().transpose()
            )

            # re-format the dataframe for plotting
            plot_df = mean_windowed_df.rename(
                columns=mean_windowed_df.loc[self.run_name]
            ).drop(index=self.run_name)

        else:  # aggregate across entire walking sessions
            title_suffix = "across_sessions"

            # get data for the chosen subject
            across_sessions_df = self.across_session_features[
                self.across_session_features["sub"] == sub
            ].transpose()

            # re-format the dataframe for plotting
            plot_df = across_sessions_df.rename(
                columns=across_sessions_df.loc[self.run_name]
            ).drop(index=[self.run_name, "sub"])

        plot_df = plot_df.apply(
            pd.to_numeric, errors="ignore"
        )  # convert column data type from object to float

        # normalize by the first visit
        plot_df.reset_index(
            inplace=True
        )  # reset index to keep the gait parameter names
        plot_df.rename_axis(
            columns=self.run_name, inplace=True
        )  # rename column name of the index for plotting (so we can color by self.run_name)
        plot_df = plot_df.loc[plot_df["index"].str[-1].argsort()]

        # normalize columns by values in the first run
        run_cols_df = plot_df.filter(
            regex="1$|2$"
        )  # filter columns that end with "1" or "2"
        run_col_1 = run_cols_df.columns[run_cols_df.columns.str.endswith("1")][
            0
        ]  # Get the first column name that ends with "1"
        run_cols_df_norm = run_cols_df.apply(
            lambda x: x / x[run_col_1], axis=1
        )  # normalize by the column
        run_cols_df_norm.columns = [
            col + "_norm" for col in run_cols_df.columns
        ]  # # Rename columns with "_norm" suffix
        plot_df_norm = pd.concat(
            [plot_df, run_cols_df_norm], axis=1
        )  # # Concatenate the original DataFrame and the normalized DataFrame

        melt_df = pd.melt(
            plot_df_norm, id_vars="index", value_vars=run_cols_df_norm.columns.values
        )

        melt_df.rename(columns={"value": "norm_value"}, inplace=True)
        melt_df["abs_value"] = np.round(
            np.concatenate(
                (run_cols_df.iloc[:, 0].values, run_cols_df.iloc[:, 1].values),
                axis=None,
            ),
            6,
        )

        # sort the gait parameters by name (in reversed order, start with the end of the string)
        melt_df["index_reverse"] = melt_df["index"].apply(self.reverse_string)
        melt_df_sorted = melt_df.sort_values(by=["index_reverse", "variable"]).drop(
            columns=["index_reverse"]
        )

        fig = px.line_polar(
            melt_df_sorted,
            r="norm_value",
            theta="index",
            color="variable",
            hover_data=["abs_value"],
            line_close=True,
            title=f"Normalized Gait Parameters {sub} {title_suffix}",
            width=700,
            height=600,
        )
        fig.update_layout(margin=dict(l=120, r=150, t=60, b=40))

        if save_fig:
            fig.write_image(
                os.path.join(
                    self.data_base_path,
                    "processed",
                    "figures_radar_plot",
                    f"radar_plot_{title_suffix}_{sub}.png",
                )
            )
        fig.show()

    def radar_plot(self, sub, by_window, save_fig):
        """Make radar plot for the chosen subject visits 1 and 2 with matplotlib

        Args:
            sub (str): subject
            by_window (Boolean): whether to use windowed features or
                aggregated features across the entire walking session
            save_fig (Boolean): whether to save the figure
        """

        if by_window:  # aggregate by windows
            title_suffix = "by_window"

            sub_windowed_df = self.windowed_features[
                self.windowed_features["sub"] == sub
            ].copy()
            mean_windowed_df = (
                sub_windowed_df.groupby(self.run_name).mean().reset_index().transpose()
            )

            # re-name columns for plotting
            plot_df = mean_windowed_df.rename(
                columns=mean_windowed_df.loc[self.run_name]
            ).drop(index=self.run_name)

        else:  # aggregate across entire walking sessions
            title_suffix = "across_sessions"

            # get data for the chosen subject
            across_sessions_df = (
                self.across_session_features[self.across_session_features["sub"] == sub]
                .copy()
                .transpose()
            )

            # re-name columns for plotting
            plot_df = across_sessions_df.rename(
                columns=across_sessions_df.loc[self.run_name]
            ).drop(index=[self.run_name, "sub"])

        # add reference data from control subjects at the end of the dataframe
        plot_df["healthy_controls"] = pd.concat(
            [plot_df, self.load_all_ref_subs()], axis=1
        )[0]

        # add data from the left and right foot
        for foot in ["left", "right"]:
            # get data for the chosen side
            foot_df = (
                self.across_session_features_LR[foot][
                    self.across_session_features["sub"] == sub
                ]
                .copy()
                .transpose()
            )

            # re-name columns for plotting the visits, remove sub name
            foot_df = foot_df.rename(columns=foot_df.loc[self.run_name]).drop(
                index=[self.run_name, "sub"]
            )

            # add reference data from control subjects at the end of the dataframe
            foot_df["healthy_controls"] = pd.concat(
                [foot_df, self.load_all_ref_subs()], axis=1
            )[0]

            # add foot name in all gait parameter names
            foot_df.set_index(foot_df.index.astype(str) + f"_{foot}", inplace=True)

            # append data from the left and right foot
            plot_df = plot_df.append(foot_df, ignore_index=False)

        plot_df.reset_index(
            inplace=True
        )  # reset index to keep the gait parameter names

        # get only the values to be plotted as circles in the radar plot
        run_cols_df = plot_df.drop(columns=["index"])

        # # normalize columns by values in the first run
        # run_col_1 = run_cols_df.columns[run_cols_df.columns.str.endswith("1")][
        #     0
        # ]  # Get the first column name that ends with "1"

        run_cols_df_norm = run_cols_df.apply(
            # lambda x: x / x[run_col_1], axis=1    # normalize by run 1
            lambda x: x / x["healthy_controls"],
            axis=1,  # normalize by healthy controls
        )  # normalize by the column
        run_cols_df_norm.columns = [
            col + "_norm" for col in run_cols_df.columns
        ]  # # Rename columns with "_norm" suffix

        plot_df = pd.concat(
            [plot_df, run_cols_df_norm], axis=1
        ).copy()  # # Concatenate the original DataFrame and the normalized DataFrame

        # remove clearance parameters, the large values make the plot unreadable
        plot_df = plot_df[~plot_df["index"].str.contains("clearance")]

        feature_subset_list = [
            ("Average Left", "avg_left"),
            ("Average Right", "avg_right"),
            ("Symmetry", "SI"),
            ("Variation", "CV"),
        ]

        # get the largest values for left and right average to set ylim
        max_avg = (
            plot_df.filter(like="norm")[plot_df["index"].str.contains("avg_")]
            .max()
            .max()
        )

        # create a figure to inset the subplots
        nrows = 2
        ncols = 2
        fig = plt.figure(figsize=(9, 9))
        for i, feature_subset in enumerate(feature_subset_list):
            # get feature subset and convert data to arrays
            runs_plot = None
            if feature_subset[0] == "All":
                variables = plot_df["index"].values
                runs_plot = plot_df.filter(regex="norm$", axis=1)
                # subtitle_suffix = f"(normalized by {self.run_name} 1)"  # normalize by run 1
                subtitle_suffix = (
                    f"(normalized by healthy controls)"  # normalize by healthy controls
                )
                subset_plot_df = plot_df.copy()
            else:
                subset_idx = plot_df["index"].str.endswith(feature_subset[1])
                variables = plot_df.loc[subset_idx, "index"].values
                subset_plot_df = plot_df[subset_idx].copy()  # get the feature subset

                # remove the suffix from variables for plotting
                pattern = re.compile(
                    r"(_avg|_avg_left|_avg_right|_SI|_CV)$"
                )  # Define pattern to match suffixes
                for j in range(len(variables)):  # Remove suffixes from each list item
                    variables[j] = re.sub(pattern, "", variables[j])

                # rename variables for plotting
                variables_for_plotting = [
                    " ".join(variable.split("_")).title() for variable in variables
                ]

                if "avg" in feature_subset[1]:  # normalize average values
                    runs_plot = subset_plot_df.filter(regex="norm$", axis=1)
                    # subtitle_suffix = f"(normalized by {self.run_name} 1)"  # normalize by run 1
                    subtitle_suffix = f"(normalized by healthy controls)"  # normalize by healthy controls

                else:  # do not normalize SI and CV, take the original values
                    runs_plot = subset_plot_df.loc[:, run_cols_df.columns.values].copy()
                    subtitle_suffix = "(original values)"

                # take the absolute values for SI
                if feature_subset[1] == "SI":
                    runs_plot = runs_plot.abs()

            # calculate angles for each variable
            angles = np.linspace(0, 2 * np.pi, len(variables), endpoint=False)

            # close the plot by appending the first value to the end
            angles = np.concatenate((angles, [angles[0]]))

            # create the radar plot
            ax = fig.add_subplot(nrows, ncols, i + 1, polar=True)
            for col_name in runs_plot:
                runs_plot_circle = np.append(
                    runs_plot[col_name].values, runs_plot[col_name].values[0]
                )  # close the plot by appending the first value to the end
                ax.plot(
                    angles,
                    runs_plot_circle,
                    "o-",
                    linewidth=1.5,
                    markersize=3,
                    label=" ".join(col_name.split("_")).title(),
                )
                ax.fill(angles, runs_plot_circle, alpha=0.25)
            ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels=variables_for_plotting)

            # display a maximum of 4 y-ticks
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

            # Adjust the position of each tick label
            for label, angle in zip(ax.get_xticklabels(), angles):
                if angle in [0, np.pi]:
                    label.set_verticalalignment("center")
                elif 0 < angle < np.pi:
                    label.set_verticalalignment("bottom")
                else:
                    label.set_verticalalignment("top")
                if angle < 0.5 * np.pi or angle > 1.5 * np.pi:
                    label.set_horizontalalignment("left")
                else:
                    label.set_horizontalalignment("right")

            ax.tick_params(axis="x", which="major", pad=-10)

            if feature_subset[1] == "SI":
                # set the lower ylim for SI by adding fake data at -0.1
                ax.plot(angles, [-0.1] * len(runs_plot_circle), "--", alpha=0)
                # draw circle at 0 for reference
                ax.plot(angles, [0] * len(runs_plot_circle), "--", color="g")

            if "avg" in feature_subset[1]:
                # set y limit to the largest value
                ax.set_ylim(0, max_avg * 1.1)

            ax.set_title(f"{feature_subset[0]} {subtitle_suffix}", y=1.15)

        ax.grid(True)

        # # adjust space around subplots
        fig.tight_layout(h_pad=0, w_pad=3)  #
        # plt.subplots_adjust(top=0.85)  # adjust the suptitle

        # # Adjust subplots to create space for legend
        # plt.subplots_adjust(right=0.8)

        # # add super title
        # sub_all_strides_df = self.all_strides_df[self.all_strides_df["sub"] == sub]
        # plt.suptitle(
        #     f"Radar plots {sub} {title_suffix} \n Total num. of strides: {self.count_num_per_run(sub_all_strides_df)}",
        #     fontsize=14,
        # )

        #  Add common legend outside the subplots using the legend of the last subplot
        plt.legend(loc="upper right", bbox_to_anchor=(-0.1, 1.6), borderaxespad=0.0)

        if save_fig:
            plt.savefig(
                os.path.join(
                    self.data_base_path,
                    "processed",
                    "figures_radar_plot",
                    f"radar_plot_healthy_speed2_{title_suffix}_{sub}.pdf",
                )
            )

        # plt.show()

    def boxplot_windows(self, sub, gait_parameter, save_fig):
        """Make boxplot for windowed strides, visualize the data distribution in all windows
           and compare with the entire walking session

        Args:
            sub (str): subject
            gait_parameter (str): gait parameter of interest
            save_fig (Boolean): whether to save the figure or not
        """

        sub_windowed_features = self.windowed_features[
            self.windowed_features["sub"] == sub
        ].copy()

        sub_all_strides_df = self.all_strides_df[
            self.all_strides_df["sub"] == sub
        ].copy()
        sub_all_strides_df["window_num"] = "all"

        sub_windowed_all_stride_df = self.windowed_all_strides_df[
            self.windowed_all_strides_df["sub"] == sub
        ].copy()

        sub_df = pd.concat([sub_all_strides_df, sub_windowed_all_stride_df], axis=0)

        # create color palette to distinguish all data and windowed data
        unique_windows = sub_df["window_num"].unique()
        colors = {
            val: "coral" if val == "all" else "turquoise" for val in unique_windows
        }
        palette = [colors[val] for val in unique_windows]

        # make boxplots of all windows for left-and right foot
        fig = plt.figure(figsize=(12, 5))
        ax = sns.boxplot(
            x=self.run_name,
            y=gait_parameter,
            hue=sub_df["window_num"],
            data=sub_df,
            palette=palette,
        )

        # Create a custom legend
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="0.25")
            for color in palette
        ]
        labels = ["all", "windows"]
        plt.legend(handles[:2], labels)

        plt.title(
            f"Boxplot of strides for Sub {sub} {gait_parameter}\n \
            Total num. of valid strides: visit1 = {sub_all_strides_df[sub_all_strides_df['visit'] == 'visit1'].shape[0]}, visit2 = {sub_all_strides_df[sub_all_strides_df['visit'] == 'visit2'].shape[0]} \n \
            Num. of windows: visit1 = {sub_windowed_features[sub_windowed_features['visit'] == 'visit1'].shape[0]}, visit2 = {sub_windowed_features[sub_windowed_features['visit'] == 'visit2'].shape[0]}"
        )

        if save_fig:
            plt.savefig(
                os.path.join(
                    self.data_base_path,
                    "processed",
                    "figures_compare_windowing",
                    f"{gait_parameter}_boxplot_{sub}.png",
                )
            )
        # plt.show()
        # plt.close()

    def scatter_plot_strides(self, sub, gait_parameter, save_fig):
        """Make scatter plot for all valid strides

        Args:
            sub (str): subject
            gait_parameter (str): gait parameter of interest
            save_fig (Boolean): whether to save the figure or not
        """

        # get data for the subject
        sub_all_strides_df = self.all_strides_df[
            self.all_strides_df["sub"] == sub
        ].copy()

        fig = plt.figure(figsize=(12, 5))
        for run in self.run_list:
            for foot_name in ["left", "right"]:
                subset_plot_df = sub_all_strides_df[
                    (sub_all_strides_df[self.run_name] == run)
                    & (sub_all_strides_df["foot"] == foot_name)
                ].copy()
                n_strides = subset_plot_df.shape[0]
                plt.plot(
                    subset_plot_df["timestamp"],
                    subset_plot_df[gait_parameter],
                    linestyle="None",
                    marker="o",
                    label=f"{run} {foot_name} n={n_strides}",
                )
                plt.plot(  # plot turning intervals
                    subset_plot_df[subset_plot_df["turning_interval"] == True][
                        "timestamp"
                    ],
                    subset_plot_df[subset_plot_df["turning_interval"] == True][
                        gait_parameter
                    ],
                    linestyle="None",
                    marker="o",
                    markersize=9,
                    markerfacecolor="None",
                    markeredgecolor="red",
                    label=f"{run} {foot_name} turning interval",
                )
        plt.xlabel("Time [s]")
        plt.title(
            f"Boxplot of strides for Sub {sub} {gait_parameter}\n \
            Total num. of valid strides:"
            + f" visit1 = {sub_all_strides_df[sub_all_strides_df['visit'] == 'visit1'].shape[0]},"
            + f" visit2 = {sub_all_strides_df[sub_all_strides_df['visit'] == 'visit2'].shape[0]}"
        )
        plt.legend()
        if save_fig:
            plt.savefig(
                os.path.join(
                    self.data_base_path,
                    "processed",
                    "figures_turning_interval",
                    f"{gait_parameter}_scatter_plot_{sub}_drop-2-stride-interval.png",
                )
            )

        plt.show()

    def plot_LR_diff(self, sub, gait_parameter, save_fig):
        # plot differences between left and right feet from windows and across entire session

        # get data for selected subject
        sub_all_strides_df = self.all_strides_df[self.all_strides_df["sub"] == sub]
        sub_windowed_all_strides_df = self.windowed_all_strides_df[
            self.windowed_all_strides_df["sub"] == sub
        ].copy()

        # loop through the two visits
        visits = ["visit1", "visit2"]
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 6), sharey=True)
        for i, ax in enumerate(axs):
            sub_all_strides_df_visit = sub_all_strides_df[
                sub_all_strides_df[self.run_name] == visits[i]
            ].copy()
            sub_windowed_all_strides_df_visit = sub_windowed_all_strides_df[
                sub_windowed_all_strides_df[self.run_name] == visits[i]
            ].copy()

            # get average values from left and right
            all_strides_LR = {}
            windowed_all_strides_LR = {}
            for foot in ["left", "right"]:
                all_strides_LR[foot] = sub_all_strides_df_visit[
                    sub_all_strides_df_visit["foot"] == foot
                ].copy()
                windowed_all_strides_LR[foot] = sub_windowed_all_strides_df_visit[
                    sub_windowed_all_strides_df_visit["foot"] == foot
                ].copy()

            # calculate the difference: left - right
            all_strides_LR_diff = (
                all_strides_LR["left"].loc[:, gait_parameter].mean()
                - all_strides_LR["right"].loc[:, gait_parameter].mean()
            )
            windowed_mean_L = (
                windowed_all_strides_LR["left"]
                .groupby(["window_num", "foot"])
                .mean()[gait_parameter]
                .values
            )
            windowed_mean_R = (
                windowed_all_strides_LR["right"]
                .groupby(["window_num", "foot"])
                .mean()[gait_parameter]
                .values
            )
            windowed_all_strides_LR_diff = windowed_mean_L - windowed_mean_R

            # boxplot of the windowed diff
            bp = ax.boxplot(windowed_all_strides_LR_diff)
            # box_x = bp['boxes'][0].get_xdata()
            ax.scatter(
                np.linspace(0.6, 0.8, len(windowed_all_strides_LR_diff)),
                windowed_all_strides_LR_diff,
                color="coral",
                s=5,
            )
            mean = np.mean(windowed_all_strides_LR_diff)
            ax.axhline(y=mean, color="coral", linestyle="-", label="Mean of windows")
            ax.axhline(
                y=all_strides_LR_diff,
                color="turquoise",
                linestyle="-",
                label="Mean of all strides",
            )
            ax.set_xticks([])  # hide x ticks
            ax.set_xlabel(visits[i])
            ax.set_ylabel(f"{gait_parameter} diff: left - right")
            ax.legend()
            ax.set_title(
                f"Total num. of valid strides = {sub_all_strides_df_visit.shape[0]} \n Num. of windows = {len(windowed_all_strides_LR_diff)}"
            )

        plt.suptitle(f"Boxplot of diff: left - right for {sub} {gait_parameter}")

        if save_fig:
            plt.savefig(
                os.path.join(
                    self.data_base_path,
                    "processed",
                    "figures_compare_windowing",
                    f"{sub}_{gait_parameter}_windowed_LR_diff_boxplot.png",
                )
            )

        # plt.show()
