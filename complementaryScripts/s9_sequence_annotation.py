#!/usr/bin/python
# coding: utf-8
# Date: 2024-05-06

import json
import numpy as np
import pandas as pd
from os.path import join


def mutant_process(sequence, mutantSites) :
    mutatedSeq = sequence
    for mutantSite in mutantSites :
        # print(mutantSite)
        # print(mutatedSeq[int(mutantSite[1:-1])-1])
        # print(mutantSite[0])
        # print(mutantSite[-1])
        if int(mutantSite[1:-1])-1 < len(mutatedSeq) :
            if mutatedSeq[int(mutantSite[1:-1])-1] == mutantSite[0] :
                mutatedSeq = list(mutatedSeq)
                mutatedSeq[int(mutantSite[1:-1])-1] = mutantSite[-1]
                mutatedSeq = ''.join(mutatedSeq)
            else :
                mutatedSeq = ''
                break
        else :
            mutatedSeq = ''
            break

    return mutatedSeq

def read_uniprot(datasets_dir, annotate_dir) :
    with open(join(datasets_dir, annotate_dir, "uniprot_seq.json"), "r") as infile :
        uniprot_seq = json.load(infile)
        return uniprot_seq

def add_sequence(datasets_dir, annotate_dir, typeFile) :
    uniprot_seq = read_uniprot(datasets_dir, annotate_dir)
    data_df_kinetics = pd.read_csv(join(datasets_dir, annotate_dir, "data_df_%s_SMILES_UNIPROT.csv" % typeFile))
    print("Entries with SMILES and Uniprot info:", len(data_df_kinetics))
    data_df_kinetics["Sequence"] = ""
    for index, row in data_df_kinetics.iterrows() :
        enzymeType = row["EnzymeType"]
        # print(enzymeType)
        protein_ID = row["Protein_ID"]
        if protein_ID in uniprot_seq.keys() :
            if enzymeType == "wildtype" :
                data_df_kinetics.at[index, 'Sequence'] = uniprot_seq[protein_ID]
            else :
                mutantSites = enzymeType.split("/")
                # print(mutantSites)
                seq = uniprot_seq[protein_ID]
                mutatedSeq = mutant_process(seq, mutantSites)
                # print(mutatedSeq)
                if mutatedSeq != "" :
                    data_df_kinetics.at[index, 'Sequence'] = mutatedSeq

    data_df_kinetics = data_df_kinetics[data_df_kinetics["Sequence"]!=""]
    data_df_kinetics.reset_index(drop=True, inplace=True)
    print("Entries after enzymeType processing:", len(data_df_kinetics))

    return data_df_kinetics

def max_kinetics_group(group, typeValue):
    max_index = group[typeValue].idxmax()
    return group.loc[max_index]
    # max_value = group[typeValue].max()
    # return group[group[typeValue] == max_value]

def output_file_without_assay_conditions(datasets_dir, annotate_dir, data_dir) :
    categories = ["KCAT", "KM", "KCATKM"]
    categories_values = {"KCAT": "KCAT VALUE", "KM": "KM VALUE", "KCATKM": "KCAT/KM VALUE"}
    for category in categories :
        print("This is for:", category.lower())
        typeValue = categories_values[category]
        data_df_kinetics = add_sequence(datasets_dir, annotate_dir, typeFile=category)

        data_df_kinetics.drop(columns = ['UNIPROT', 'PH', 'Temperature'], inplace=True)
        new_order = ["EC", "SUBSTRATE", "ORGANISM", "Protein_ID", "EnzymeType", typeValue, "UNIT", "SMILES", "Sequence"]
        data_df_kinetics = data_df_kinetics.reindex(columns=new_order)
        n_old = len(data_df_kinetics)
        data_df_kinetics = data_df_kinetics.drop_duplicates(keep="first").reset_index(drop=True)
        print("We remove %s out of %s data points, because they are duplaictes."
              % (n_old - len(data_df_kinetics), n_old))

        n_old = len(data_df_kinetics)
        data_df_kinetics = data_df_kinetics.groupby(["SMILES", "Sequence"], as_index=False).apply(max_kinetics_group, typeValue)
        print("By grouping data points with the same substrate SMILES, protein sequence, this changes the number of data points from %s to %s." % (n_old, len(data_df_kinetics)))
        data_df_kinetics.sort_values(by="EC", inplace=True)
        data_df_kinetics["Log10_value"] = [np.log10(float(data)) for data in data_df_kinetics[typeValue]]
        data_df_kinetics.reset_index(drop=True, inplace=True)
        print("Total dataset after data cleaning:", len(data_df_kinetics))
        print("Done!")

        # Small test to validate what I did is right
        # data_df_kcat = data_df_kinetics.copy()
        # data_df_kcat = data_df_kcat.head(1000)
        # duplicates = data_df_kcat[data_df_kcat.duplicated(subset=["SMILES", "Sequence"], keep=False)]
        # groups = duplicates.groupby(["SMILES", "Sequence"])
        # for name, group in groups:
        #     if len(group['EC'].unique()) > 1:
        #         print("Group:", name)
        #         display(group.head())

        data_df_kinetics.to_csv(join(datasets_dir, data_dir, "data_%s.csv" % category), index=False)

def output_file_with_assay_conditions(datasets_dir, annotate_dir, data_dir) :
    categories = ["KCAT", "KM", "KCATKM"]
    categories_values = {"KCAT": "KCAT VALUE", "KM": "KM VALUE", "KCATKM": "KCAT/KM VALUE"}
    for category in categories :
        print("This is for:", category.lower())
        typeValue = categories_values[category]
        data_df_kinetics = add_sequence(datasets_dir, annotate_dir, typeFile=category)

        data_df_kinetics.drop(columns = ['UNIPROT'], inplace=True)
        data_df_kinetics = data_df_kinetics[data_df_kinetics['PH'].notna() & data_df_kinetics['Temperature'].notna()]
        print("We have %s entries with both PH and temperature data" % len(data_df_kinetics))
        new_order = ["EC", "SUBSTRATE", "ORGANISM", "Protein_ID", "EnzymeType", "PH", "Temperature", typeValue, "UNIT", "SMILES", "Sequence"]
        data_df_kinetics = data_df_kinetics.reindex(columns=new_order)
        n_old = len(data_df_kinetics)
        data_df_kinetics = data_df_kinetics.drop_duplicates(keep="first").reset_index(drop=True)
        print("We remove %s out of %s data points, because they are duplaictes."
              % (n_old - len(data_df_kinetics), n_old))

        n_old = len(data_df_kinetics)
        data_df_kinetics = data_df_kinetics.groupby(["PH", "Temperature", "SMILES", "Sequence"], as_index=False).apply(max_kinetics_group, typeValue)
        print("By grouping data points with the same substrate SMILES, protein sequence and assay conditions, this changes the number of data points from %s to %s." % (n_old, len(data_df_kinetics)))
        data_df_kinetics.sort_values(by="EC", inplace=True)
        data_df_kinetics["Log10_value"] = [np.log10(float(data)) for data in data_df_kinetics[typeValue]]
        data_df_kinetics.reset_index(drop=True, inplace=True)
        print("Total dataset after data cleaning:", len(data_df_kinetics))
        print("Done!")

        data_df_kinetics.to_csv(join(datasets_dir, data_dir, "data_%s_assay_conditions.csv" % category), index=False)

def main() :
    datasets_dir = "../complementaryData"
    annotate_dir = "data_annotation"
    data_dir = "data"
    output_file_without_assay_conditions(datasets_dir, annotate_dir, data_dir)
    output_file_with_assay_conditions(datasets_dir, annotate_dir, data_dir)


if __name__ == '__main__' :
    main()


