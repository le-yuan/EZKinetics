#!/usr/bin/python
# coding: utf-8
# Date: 2026-04-07

import os
import pandas as pd
from os.path import join
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def load_prediction_file(prediction_path):
    df = pd.read_csv(prediction_path)
    return df

def calculate_metrics(prediction_path):
    df = load_prediction_file(prediction_path)
    y_test = df["y_test"].values
    y_pred = df["y_pred"].values

    r2 = r2_score(y_test, y_pred)
    pcc, _ = pearsonr(y_test, y_pred)
    return r2, pcc

def parse_hyperparams(folder_name):
    parts = folder_name.split("_")
    lr = float(parts[1])
    batch_size = int(parts[3])
    dropout = float(parts[5])
    return lr, batch_size, dropout

def collect_results(model_root):
    results = []
    for subdir in os.listdir(model_root):
        subdir_path = join(model_root, subdir)
        prediction_file = join(subdir_path, "test_data_predictions.csv")

        r2, pcc = calculate_metrics(prediction_file)
        lr, batch_size, dropout = parse_hyperparams(subdir)

        results.append({
            "Model": "FFNN",
            "batch_size": batch_size,
            "lr": lr,
            "dropout": dropout,
            "R2": r2,
            "PCC": pcc
        })

    return pd.DataFrame(results)


def main():
    model_root = "../model/KCAT_KM/FFNN/ESM2_650M&ChemBERTa-77M-MTR"
    save_path = "../data/analysis"
    os.makedirs(save_path, exist_ok=True)

    df = collect_results(model_root)
    df = df.sort_values(by=["lr", "batch_size", "dropout"])
    df["R2"] = df["R2"].round(4)
    df["PCC"] = df["PCC"].round(4)

    output_file = join(save_path, "FFNN_results.csv")
    df.to_csv(output_file, index=False)
    print("Saved to:", output_file)


if __name__ == "__main__":
    main()


