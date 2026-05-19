#!/usr/bin/python
# coding: utf-8
# Date: 2024-04-29

import json
import numpy as np
import pandas as pd
from rdkit import Chem
from os.path import join


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    return canonical_smiles

def max_kinetics_group(group, typeValue):
    max_index = group[typeValue].idxmax()
    return group.loc[max_index]
    # max_value = group[typeValue].max()
    # return group[group[typeValue] == max_value]

def main() :
    datasets_dir = "../complementaryData"
    combine_dir = "data_combination"
    annotate_dir = "data_annotation"

    with open(join(datasets_dir, combine_dir, "substrate_smiles.json"), "r") as infile :
        substrate_smiles = json.load(infile)

    substrate_canonical_smiles = dict()
    for substrate, smiles in substrate_smiles.items() :
        canonical_smiles = canonicalize_smiles(smiles)
        # print(canonical_smiles)
        if smiles is not None :
            substrate_canonical_smiles[substrate] = canonical_smiles
    print("Substrates with canonical SMILES:", len(substrate_canonical_smiles))  # 27438

    categories = ["KCAT", "KM", "KCATKM"]
    categories_values = {"KCAT": "KCAT VALUE", "KM": "KM VALUE", "KCATKM": "KCAT/KM VALUE"}
    for category in categories :
        print("This is for:", category.lower())
        data_df_kinetics = pd.read_csv(join(datasets_dir, combine_dir, "data_df_%s.csv" % category))
        data_df_kinetics["SMILES"] = ""
        for index, row in data_df_kinetics.iterrows():
            substrate = row['SUBSTRATE']
            if substrate in substrate_canonical_smiles.keys() :
                data_df_kinetics.at[index, 'SMILES'] = substrate_canonical_smiles[substrate]

        n_old = len(data_df_kinetics)
        typeValue = categories_values[category]
        columns_to_replace = ["UNIPROT", "PH", "Temperature"]
        data_df_kinetics[columns_to_replace] = data_df_kinetics[columns_to_replace].replace(np.nan, "")
        data_df_kinetics = data_df_kinetics.groupby(["EC", "ORGANISM", "UNIPROT", "EnzymeType", "PH", "Temperature", "UNIT", "SMILES"]).apply(max_kinetics_group, typeValue)
        print("By grouping data points with same EC number, substrate SMILES, organism and UniProt ID, this changes the number of data points from %s to %s." % (n_old, len(data_df_kinetics)))
        data_df_kinetics.reset_index(drop=True, inplace=True)
        print("Done!")
        # data_df_kinetics

        # write output file
        data_df_kinetics.to_csv(join(datasets_dir, annotate_dir, "data_df_%s_SMILES.csv" % category), index=False)

        # # Small test I, it can work:
        # def max_kcat_group(group, typeValue):
        #     max_index = group[typeValue].idxmax()
        #     return group.loc[max_index]

        # import numpy as np
        # data_df = data_df_kinetics.copy()
        # n_old = len(data_df)
        # typeValue = "KCAT VALUE"
        # columns_to_replace = ["UNIPROT", "PH", "Temperature"]
        # data_df[columns_to_replace] = data_df[columns_to_replace].replace(np.nan, "")
        # data_df = data_df.groupby(["EC", "ORGANISM", "UNIPROT", "EnzymeType", "PH", "Temperature", "UNIT", "SMILES"]).apply(max_kcat_group, typeValue)
        # print("By grouping data points with same EC number, substrate SMILES, organism and UniProt ID, this changes the number of data points from %s to %s." % (n_old, len(data_df)))
        # data_df.reset_index(drop=True, inplace=True)
        # print("Done!")
        # data_df

        # # Small test II, it can also work, all the three columns ["UNIPROT", "PH", "Temperature"] are string format:
        # def max_kcat_group(group, typeValue):
        #     max_index = group[typeValue].idxmax()
        #     return group.loc[max_index]

        # data_df = data_df_kinetics.copy()
        # columns_to_replace = ["UNIPROT", "PH", "Temperature"]
        # data_df[columns_to_replace] = data_df[columns_to_replace].astype(str)
        # n_old = len(data_df)
        # typeValue = "KCAT VALUE"
        # # data_df["UNIPROT"].fillna("", inplace=True)
        # # data_df["PH"].fillna("", inplace=True)
        # # data_df["Temperature"].fillna("", inplace=True)
        # data_df = data_df.groupby(["EC", "ORGANISM", "UNIPROT", "EnzymeType", "PH", "Temperature", "UNIT", "SMILES"]).apply(max_kcat_group, typeValue)
        # print("By grouping data points with same EC number, substrate SMILES, organism and UniProt ID, this changes the number of data points from %s to %s." % (n_old, len(data_df)))
        # data_df.reset_index(drop=True, inplace=True)
        # print("Done!")
        # data_df


if __name__ == '__main__' :
    main()


