#!/usr/bin/python
# coding: utf-8
# Date: 2024-07-19

import os
import pandas as pd
from os.path import join
from sklearn.model_selection import train_test_split


def main() :
    datasets_dir = "../data"
    sub_dir = "kinetics_data"
    typeFiles = ["KCATKM", "KCAT", "KM"]

    for typeFile in typeFiles :
        data_df_kinetics = pd.read_csv(join(datasets_dir, sub_dir, "data_%s.csv" % typeFile))
        train_df, test_df = train_test_split(data_df_kinetics, test_size=0.2, random_state=42)
        # print(train_df.dtypes)

        if typeFile == "KCATKM" :
            if not os.path.exists(join(datasets_dir, "KCAT_KM", "train_test")) :
                os.makedirs(join(datasets_dir, "KCAT_KM", "train_test"), exist_ok=True)

            # Save train and test dataset
            train_df.to_csv(join(datasets_dir, "KCAT_KM", "train_test", 'train.csv'), index=False)
            test_df.to_csv(join(datasets_dir, "KCAT_KM", "train_test", 'test.csv'), index=False)

        if typeFile == "KCAT" :
            if not os.path.exists(join(datasets_dir, "KCAT", "train_test")) :
                os.makedirs(join(datasets_dir, "KCAT", "train_test"), exist_ok=True)

            # Save train and test dataset
            train_df.to_csv(join(datasets_dir, "KCAT", "train_test", 'train.csv'), index=False)
            test_df.to_csv(join(datasets_dir, "KCAT", "train_test", 'test.csv'), index=False)

        if typeFile == "KM" :
            if not os.path.exists(join(datasets_dir, "KM", "train_test")) :
                os.makedirs(join(datasets_dir, "KM", "train_test"), exist_ok=True)

            # Save train and test dataset
            train_df.to_csv(join(datasets_dir, "KM", "train_test", 'train.csv'), index=False)
            test_df.to_csv(join(datasets_dir, "KM", "train_test", 'test.csv'), index=False)


if __name__ == '__main__' :
    main()


