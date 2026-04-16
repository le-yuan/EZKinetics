#!/usr/bin/python
# coding: utf-8
# Date: 2025-05-07

import os
import pandas as pd
from os.path import join


def data_summarize(datasets_dir, data_dir, typeFile):
    data_df = pd.read_csv(join(datasets_dir, data_dir, f"data_{typeFile}.csv"))

    return {
        "Total entries": len(data_df),
        "Unique EC Number": data_df["EC"].nunique(),
        "Unique substrates": data_df["SMILES"].nunique(),
        "Unique organisms": data_df["ORGANISM"].nunique(),
        "Unique sequences": data_df["Sequence"].nunique(),
    }

def main():
    datasets_dir = "../data"
    data_dir = "kinetics_data"
    output_dir = join(datasets_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    categories = ["KCAT", "KM", "KCATKM"]
    results = {}
    for category in categories:
        results[category] = data_summarize(datasets_dir, data_dir, category)

    df_summary = pd.DataFrame(results)
    desired_order = [
        "Total entries",
        "Unique EC Number",
        "Unique substrates",
        "Unique organisms",
        "Unique sequences",
    ]

    df_summary = df_summary.reindex(desired_order)
    output_path = join(output_dir, "data_summary.csv")
    df_summary.to_csv(output_path)
    print(df_summary)


if __name__ == '__main__' :
    main()

    # Results:
    
    # Processing: KCAT
    # Total entries: 33376
    # Unique EC Number: 2437
    # Unique substrates: 5765
    # Unique organisms: 1670
    # Unique sequences: 12817


    # Processing: KM
    # Total entries: 50303
    # Unique EC Number: 3240
    # Unique substrates: 7549
    # Unique organisms: 2200
    # Unique sequences: 17752


    # Processing: KCATKM
    # Total entries: 26431
    # Unique EC Number: 2058
    # Unique substrates: 4536
    # Unique organisms: 1382
    # Unique sequences: 10124




