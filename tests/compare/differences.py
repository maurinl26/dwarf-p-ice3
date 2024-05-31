# -*- coding: utf-8 -*-
import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv("output/ice_adjust/compare.csv", header=0)
    print(df.shape)
    print(df.columns)

    df_diff = df[df["Set"] == "Relative difference"]

    df_diff.to_csv("output/ice_adjust/diff.csv")

    # df.reset_index(inplace=True)

    # run_df = df[df["Set"] == "Run"]
    # ref_df = df[df["Set"] == "Ref"]

    # print(run_df.head())
    # print(ref_df.head())
    # print(run_df.columns)
    # print(ref_df.columns)

    # run_df["Diff"] = run_df["max"] - ref_df["max"]
