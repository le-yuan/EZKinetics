#!/usr/bin/python
# coding: utf-8
# Date: 2024-04-24

import json
import requests
import warnings
import pandas as pd
from os.path import join
from py2opsin import py2opsin


# source: https://github.com/samgoldman97/brenda-parser
def process_brenda(datasets_dir, smiles_dir, file) :
    with open(join(datasets_dir, smiles_dir, file), "r") as infile1 :
        names_smiles_1 = json.load(infile1)

    print(len(names_smiles_1))  # 156323
    return names_smiles_1

# # source: https://github.com/ranaabarghout/Protein_Kinetics_Data_Scrapping/blob/main/X_DataProcessing/X00_enzyme_datasets/cmpd_smls_all.csv
def process_kinetics(datasets_dir, smiles_dir, file) :
    with open(join(datasets_dir, smiles_dir, file), "r") as infile2 :
        lines = infile2.readlines()

    names_smiles_2 = dict()
    for line in lines :
        data = line.strip().split(";")
        if data[2] != "None" :
            names_smiles_2[data[1]] = data[2]

    print(len(names_smiles_2))  # 13238
    return names_smiles_2

# source: https://github.com/maranasgroup/CatPred-DB/tree/main/datasets
def process_catpred(datasets_dir, smiles_dir, file) :
    with open(join(datasets_dir, smiles_dir, file), "r") as infile3 :
        lines = infile3.readlines()

    names_smiles_3 = dict()
    for line in lines :
        data = line.strip().split("\t")
        if len(data) == 5 :
            smiles = data[4]
            names_smiles_3[data[1]] = smiles

    print(len(names_smiles_3))  # 116294
    return names_smiles_3

def read_csv(datasets_dir, data_dir, category) :
    data_df = pd.read_csv(join(datasets_dir, data_dir, "data_df_%s.csv" % category))
    # substrate = data_df.head(100)["SUBSTRATE"].unique()
    substrate = data_df["SUBSTRATE"].unique().tolist()
    # print(len(substrate))
    # print(substrate[:5])

    return substrate

# obtain SMILES by PubChem API using the website
def get_smiles_pubchem(name):
    try :
        url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/CanonicalSMILES/TXT' % name
        req = requests.get(url)
        if req.status_code != 200:
            smiles = None
        else:
            smiles = req.content.splitlines()[0].decode()
            print(smiles)
        # redis_cli.set(name, smiles, ex=None)

        # print smiles
    except :
        smiles = None

    return smiles

def get_smiles_py2opsin(name) :
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    try :
        smiles = py2opsin(chemical_name = name, output_format = "SMILES")
        # smiles = py2opsin(name)
        if smiles != "" :
            print("py2opsin:", smiles)
        else :
            smiles = None
    except :
        smiles = None

    return smiles

def main() :
    datasets_dir = "../complementaryData"
    smiles_dir = "smiles_data"
    data_dir = "data_combination"

    names_smiles_1 = process_brenda(datasets_dir, smiles_dir, file="compounds_to_smiles.json")
    names_smiles_2 = process_kinetics(datasets_dir, smiles_dir, file="cmpd_smls_all.csv")
    names_smiles_3 = process_catpred(datasets_dir, smiles_dir, file="metabolite_inchi_smiles_brenda_pubchem.tsv")
    smiles_database = {**names_smiles_1, **names_smiles_2, **names_smiles_3}
    print("Total SMILES info:", len(smiles_database)) # 171622

    categories = ["KCAT", "KM", "KCATKM"]
    substrate_list = list()
    for category in categories :
        substrate = read_csv(datasets_dir, data_dir, category)
        substrate_list += substrate
    substrate_unique = list(set(substrate_list))
    print("Unique substrates:", len(substrate_unique))  # 32984

    substrate_smiles = dict()
    substrate_no_smiles = list()
    substrate_additional = list()
    substrate_lower_smiles = {key.lower(): value for key, value in smiles_database.items()}
    for substrate in substrate_unique :
        if substrate.lower() in substrate_lower_smiles.keys() :
            substrate_smiles[substrate] = substrate_lower_smiles[substrate.lower()]
        else :
            substrate_additional.append(substrate)

    print("SMILES can be found from our in-house database:", len(substrate_smiles))  # 26479
    print("SMILES can not be found from our in-house database:", len(substrate_additional))  # 6505

    # i = 0
    # pubchem_smiles = dict()
    # py2opsin_smiles = dict()
    # for name in substrate_additional[:50] :
    #     i += 1
    #     print(i)
    #     print(name)
    #     # print(smiles)
    #     smiles = get_smiles_pubchem(name)
    #     if smiles is not None:
    #         pubchem_smiles[name] = smiles
    #     else :
    #         smiles = get_smiles_py2opsin(name)
    #         if smiles is not None:
    #             py2opsin_smiles[name] = smiles
    # print("SMILES found by PubChem:", len(pubchem_smiles))
    # print("SMILES found by Py2opsin:", len(py2opsin_smiles))

    i = 0
    additional_smiles = dict()
    for name in substrate_additional :
        i += 1
        print(i)
        print(name)
        # print(smiles)
        smiles = get_smiles_pubchem(name)
        if smiles is None:
            smiles = get_smiles_py2opsin(name)

        if smiles is not None:
            additional_smiles[name] = smiles
            # print("----"*5)
        else :
            substrate_no_smiles.append(name)

    substrate_smiles.update(additional_smiles)
    print("SMILES can be found:", len(substrate_smiles))         # 27438
    print("SMILES can not be found:", len(substrate_no_smiles))  # 5546

    with open(join(datasets_dir, data_dir, "substrate_smiles.json"), "w") as outfile1 :
        json.dump(substrate_smiles, outfile1, indent=4)
    with open(join(datasets_dir, data_dir, "substrate_no_smiles.json"), "w") as outfile2 :
        json.dump(substrate_no_smiles, outfile2, indent=4)


if __name__ == '__main__' :
    main()


