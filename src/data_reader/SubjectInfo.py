import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm


class SubjectInfo:
    """
    summarizes information about the subjects
    """

    def __init__(self, dataset, sub_list=None):
        """_summary_

        Args:
            dataset (_type_): name of the dataset (string)
        """

        with open("path.json") as f:
            paths = json.load(f)
        self.base_data_path = paths[f"data_{dataset}"]

        # read charite data
        if dataset == "charite":
            read_path = os.path.join(self.base_data_path, "raw", "subject_info.xlsx")
            self.data_df = pd.read_excel(read_path, sheet_name="med_info")
            self.data_collection_df = pd.read_excel(
                read_path, sheet_name="data_collection"
            )

            # rename columns
            self.data_df = self.data_df.rename(
                columns={
                    "sub": "id",
                    "height_cm": "height(cm)",
                    "weight_kg": "weight(kg)",
                }
            )
            self.data_collection_df = self.data_collection_df.rename(
                columns={"sub": "id"}
            )

        # read kiel data
        elif dataset == "kiel" or dataset == "kiel_val":
            read_path = os.path.join(self.base_data_path, "raw", "subject_info.csv")

            self.data_df = pd.read_csv(
                read_path
            )  # dataframe with original subject info
            self.data_df["id"] = self.data_df["id"].astype(str).str.zfill(3)
            self.data_df["id"] = "pp" + self.data_df["id"].astype(str)

            # rename columns
            self.data_df = self.data_df.rename(
                columns={
                    "gender": "sex",
                    "vital_height": "height(cm)",
                    "vital_weight": "weight(kg)",
                }
            )

        # filter subjects to be analyzed
        if sub_list != None:
            self.data_df = self.data_df[self.data_df["id"].isin(sub_list)]
            self.data_collection_df = self.data_collection_df[
                self.data_collection_df["id"].isin(sub_list)
            ]
            # reset index
            self.data_df.reset_index(inplace=True, drop=True)
            self.data_collection_df.reset_index(inplace=True, drop=True)

    def anthropometrics(self):
        """
        summary of subjects' age, height, weight, leg length
        @return: print out mean +- standard deviation
        @rtype:
        """
        # counts
        print("Data count:")
        print(self.data_df.groupby("sex").count())
        # print number of subjects with missing data in sex, age, height, weight columns
        print("number of subjects with missing data:")
        print(self.data_df[["sex", "age", "weight(kg)", "height(cm)"]].isnull().sum())
        print()

        # summary statistics
        for item in [
            "age",
            "weight(kg)",
            "height(cm)",
            "FAC_visit1",
            "FAC_visit2",
            "visit1_days_since_stroke",
            "visit2_days_since_stroke",
        ]:
            mean = round(self.data_df[item].mean(), 1)
            std = round(self.data_df[item].std(), 1)
            min = round(self.data_df[item].min(), 1)
            max = round(self.data_df[item].max(), 1)

            print(f"{item}: {mean} +- {std}, min: {min}, max: {max}")

    def improvement_evaluation(self):
        """Compare the improvement of the subjects between the two visits."""

        # Create a copy of the data and filter out the columns that are not needed
        evaluation_df = self.data_collection_df.copy()
        evaluation_df = evaluation_df[
            [
                "id",
                "self_improvement",
                "ext_improvement_1",
                "ext_improvement_2",
                "FAC_improvement",
                "gait_param_improvement",
            ]
        ]

        # prepare FAC data for plotting
        fac_df = self.data_df[["id", "FAC_visit1", "FAC_visit2"]]

        # replace id with P1, P2, ...
        for df in [evaluation_df, fac_df]:
            df.sort_values(by=["id"], inplace=True)  # sort by id
            df.set_index("id", inplace=True)  # set "id" column as index
            # df.reset_index(inplace=True)  # move id to a column
            # df.index = df.index.map(
            #     lambda x: "P" + str(x + 1)
            # )  # modify index
            # df.drop(columns=["id"], inplace=True)  # remove the id column

        # replace values for plotting
        evaluation_df = evaluation_df.replace({"Y": 1, "N": 0})  # for the color map
        annot_df = evaluation_df.replace({1: "Yes", 0: "No"})   # for the text annotation

        # replace all column names for plotting
        evaluation_df = evaluation_df.rename(
            columns={
                "self_improvement": "Self-Reported \nImprovement",
                "ext_improvement_1": "Expert 1",
                "ext_improvement_2": "Expert 2",
                "FAC_improvement": "FAC",
                "gait_param_improvement": "Our Gait \nVisualization",
            }
        )
        fac_df = fac_df.rename(
            columns={"FAC_visit1": "Visit 1", "FAC_visit2": "Visit 2"}
        )

        fig, _ = plt.subplots(figsize=(6, 6))
        gs = gridspec.GridSpec(
            1, 2, width_ratios=[evaluation_df.shape[1], fac_df.shape[1]]
        )

        # heatmap for evaluated improvements
        ax0 = plt.subplot(gs[0])
        sns.heatmap(
            evaluation_df,
            annot=annot_df,
            fmt="",
            cmap="coolwarm_r",
            cbar=False,
            ax=ax0,
        )
        ax0.set_title("Improvement Evaluation")
        ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45)
        ax0.set_yticklabels(ax0.get_yticklabels(), rotation=0)
        ax0.set_ylabel("Participant")

        # heatmap for FAC changes between visits

        ax1 = plt.subplot(gs[1])
        cmap = plt.get_cmap("viridis")
        norm = BoundaryNorm(range(1, 7), cmap.N)

        sns.heatmap(
            fac_df,
            annot=fac_df,
            cmap=cmap,
            norm=norm,
            cbar=False,  # Do not automatically create a colorbar
            ax=ax1,
            yticklabels=False,
        )
        ax1.set_title("FAC at Visits")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.set_ylabel("")

        # Create a custom colorbar with integer ticks and discrete colors
        cbar = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax1,  # Associate the colorbar with FAC
            ticks=[i + 0.5 for i in range(1, 6)],  # Set tick positions
            orientation="vertical",
            pad=0.1,  # Adjust the distance of the colorbar from the rightmost plot
        )
        # Set the tick labels to integers from 1 to 5
        cbar.ax.set_yticklabels(range(1, 6))
        cbar.outline.set_edgecolor("none")

        plt.yticks(rotation=90)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.base_data_path, "processed", "gait_improvement_evaluation.pdf"
            ),
            dpi=300,
        )
        plt.show()


#### main ####
if __name__ == "__main__":
    ## Kiel dataset ##
    control_list = [
        "pp010",
        "pp011",
        "pp028",
        # "pp071",    # part of the RF data is not usable (IMU dropped from the foot)
        "pp079",
        "pp099",
        "pp105",
        "pp106",
        # "pp114",    # only one constant treadmill speed
        "pp137",
        "pp139",
        "pp158",
        "pp165",
        # "pp166"     # only one constant treadmill speed
    ]

    stroke_list = [
        "pp077",
        "pp101",
        "pp109",
        "pp112",
        "pp122",
        "pp123",
        "pp145",
        "pp149",
    ]

    overground_list = [
        # "pp001",
        "pp002",
        "pp003",
        "pp004",
        "pp005",
        "pp006",
        # "pp007",    # l_psis is missing for the optical data
        "pp008",
        # "pp009",
        # "pp010"
    ]

    charite_list = [
        "imu0001",
        "imu0002",
        "imu0003",
        "imu0006",
        "imu0007",
        "imu0008",
        "imu0009",
        "imu0010",  # only has visit 1
        "imu0011",
        "imu0012",
        "imu0013",
        "imu0014",  # only has visit 1
    ]

    subject_info = SubjectInfo("charite", sub_list=charite_list)
    subject_info.anthropometrics()
