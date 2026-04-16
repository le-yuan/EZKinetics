#!/usr/bin/python
# coding: utf-8
# Date: 2026-01-01

import os
import subprocess
import pandas as pd
from os.path import join
from collections import defaultdict
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def save_as_fasta(df, filename, prefix):
    with open(filename, "w") as f:
        for i, row in df.iterrows():
            f.write(f">{prefix}_{i+1}\n{row['Sequence']}\n")

def run_mmseqs(test_fasta, train_fasta, result_file, tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    cmd = [
        "mmseqs", "easy-search",
        test_fasta,
        train_fasta,
        result_file,
        tmp_dir,
        "--format-output", "query,target,pident"
    ]
    subprocess.run(cmd, check=True)

def parse_mmseqs_output(path):
    results = []
    with open(path) as f:
        for line in f:
            q, t, ident = line.strip().split()
            results.append((q, t, float(ident)))
    return results

def split_by_identity(results):
    max_id = defaultdict(float)
    for q, _, ident in results:
        if ident > max_id[q]:
            max_id[q] = ident

    groups = {
        "0-40": [],
        "40-80": [],
        "80-99": [],
        "99-100": []
    }

    for k, v in max_id.items():
        if v <= 40:
            groups["0-40"].append(k)
        elif v <= 80:
            groups["40-80"].append(k)
        elif v <= 99:
            groups["80-99"].append(k)
        else:
            groups["99-100"].append(k)

    return groups

def evaluate(df):
    if len(df) == 0:
        return float("nan"), float("nan")
    r2 = r2_score(df["y_test"], df["y_pred"])
    pcc, _ = pearsonr(df["y_test"], df["y_pred"])
    return r2, pcc

def main():
    datasets_dir = "../data"
    training_dirs = ["KCAT_KM", "KCAT", "KM"]

    for training_dir in training_dirs[:1] :
    # for training_dir in training_dirs :
        seq_dir = join(datasets_dir, training_dir, "sequence")
        os.makedirs(seq_dir, exist_ok=True)

        train_df = pd.read_csv(join(datasets_dir, training_dir, "train_test", "train.csv"))
        test_df = pd.read_csv(join(datasets_dir, training_dir, "train_test", "test.csv"))

        # train_fasta = join(seq_dir, "train.fasta")
        # test_fasta = join(seq_dir, "test.fasta")

        # save_as_fasta(train_df, train_fasta, "Protein_train")
        # save_as_fasta(test_df, test_fasta, "Protein_test")

        result_file = join(seq_dir, "result.m8")
        tmp_dir = join(seq_dir, "tmp")

        # run_mmseqs(test_fasta, train_fasta, result_file, tmp_dir)

        mmseqs_results = parse_mmseqs_output(result_file)
        groups = split_by_identity(mmseqs_results)

        test_seq_map = {
            f"Protein_test_{i+1}": row["Sequence"]
            for i, row in test_df.iterrows()
        }

        # if training_dir == "KCAT":
        #     model_name = "ESM2_650M&ChemBERTa-77M-MTR"
        # else:
        #     model_name = "ESM2_3B&ChemBERTa-77M-MTR"

        model_name = "ESM2_3B&ChemBERTa-77M-MTR"
        pred_path = join(
            "../model",
            training_dir,
            "XGBoost",
            model_name,
            "test_data_predictions.csv"
        )
        pred_df = pd.read_csv(pred_path)

        print(model_name)
        for k, ids in groups.items():
            seqs = [test_seq_map[i] for i in ids if i in test_seq_map]
            sub_df = pred_df[pred_df["protein_sequence"].isin(seqs)]
            r2, pcc = evaluate(sub_df)
            print(f"{training_dir} | {k} | R2={r2:.4f} | PCC={pcc:.4f}")


if __name__ == "__main__":
    main()


