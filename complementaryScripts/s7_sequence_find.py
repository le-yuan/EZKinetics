#!/usr/bin/python
# coding: utf-8
# Date: 2024-05-01

import io
import pickle
import time
import json
import hashlib
import requests
import numpy as np
import pandas as pd
from zeep import Client
from os.path import join


def filter_smiles(datasets_dir, annotate_dir, typeFile) :
    data_df_kinetics = pd.read_csv(join(datasets_dir, annotate_dir, "data_df_%s_SMILES.csv" % typeFile))
    filtered_df = data_df_kinetics[data_df_kinetics["SMILES"].notna()]
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df

def filter_uniprot(df) :
    filtered_df = df[df["UNIPROT"].isna()]
    filtered_df = filtered_df.loc[:, ["EC", "ORGANISM"]]
    filtered_df = filtered_df.drop_duplicates()
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df

def get_EC_org() :
    datasets_dir = "../complementaryData"
    annotate_dir = "data_annotation"
    categories = ["KCAT", "KM", "KCATKM"]
    list_dfs = list()
    for category in categories :
        print("This is for:", category.lower())
        filtered_df = filter_smiles(datasets_dir, annotate_dir, typeFile=category)
        print("There are %s entries with SMILES information" % len(filtered_df))
        filtered_df = filter_uniprot(filtered_df)
        print("There are %s unique entries without UNIPROT information" % len(filtered_df))  
        list_dfs.append(filtered_df)
    # print(len(list_dfs))
    # for df in list_dfs :
    #     print(len(df))

    EC_org_df = pd.concat(list_dfs, ignore_index=True)
    EC_org_df = EC_org_df.drop_duplicates(keep = "first").reset_index(drop = True)
    print("There are %s different EC number-organism combinations without UNIPROT" %len(EC_org_df))  # 19088
    return EC_org_df

def download_data_from_BRENDA(function, parameters, error_identifier, print_error=False):
    """
    A function that downloads data from BRENDA. The data to be downloaded is specified by the arguments "function" and
    "parameters". The arguments "error_identifier" and "print_error" can be used for error identfication.
    """
    try:
        resultString = function(*parameters)
        return(resultString)
    except:
        #if download was unsuccessful, try again one more time:
        time.sleep(0.5)
        try:
            resultString = function(*parameters)
            return(resultString)
        except Exception as ex:
            if print_error:
                print("Download for %s was unsuccessfull (Type of error: %s)" % (error_identifier, ex))
                return([]) # return empty list if download was unsuccessfull

def brenda_query(datasets_dir, annotate_dir) :
    wsdl = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
    email = 'your_email@example.com'
    brenda_password = 'your_password'
    password = hashlib.sha256(brenda_password.encode("utf-8")).hexdigest()
    client = Client(wsdl)

    EC_org_df = get_EC_org()
    # EC_org_df["Protein_ID"] = np.nan
    EC_org_df["Protein_ID"] = ""
    for ind in EC_org_df.index:
        print(ind+1)
        org = EC_org_df["ORGANISM"][ind]
        EC = EC_org_df["EC"][ind]
        print(org)
        print(EC)
        if not pd.isnull(EC) and not pd.isnull(org):
            parameters = (email,password,"ecNumber*" + str(EC), "sequence*", "noOfAminoAcids*",
                          "firstAccessionCode*", "source*", "id*", "organism*" + str(org))

            resultString = download_data_from_BRENDA(function = client.service.getSequence,
                                                parameters = parameters, error_identifier = EC +";" +org,
                                               print_error = True)
            if resultString != [] and resultString is not None:
                EC_org_df["Protein_ID"][ind] = (',').join([resultString[i]["firstAccessionCode"] for i in range(len(resultString))])
        time.sleep(0.5)
        
    EC_org_df.to_pickle(join(datasets_dir, annotate_dir, "EC_org_df_with_sequences_from_brenda.pkl"))

def uniprot_query(datasets_dir, annotate_dir) :
    EC_org_df = pd.read_pickle(join(datasets_dir, annotate_dir, "EC_org_df_with_sequences_from_brenda.pkl"))
    data_noseq = EC_org_df.loc[EC_org_df["Protein_ID"]==""]
    data_noseq.reset_index(drop=True, inplace=True)
    print(len(data_noseq))  # 9388

    for ind in data_noseq.index :
        print(ind+1)
        # time.sleep(0.5)
        if (ind+1) % 1000 == 0 :
            time.sleep(30)
        org = data_noseq["ORGANISM"][ind]
        EC = data_noseq["EC"][ind]
        print(org)
        print(EC)
        url = f"https://rest.uniprot.org/uniprotkb/search?&query=ec:{EC}%20AND%20organism_name:{org}&format=fasta"
        # print(url)
        response = requests.get(url)
        if response.status_code == 200 :
            data = response.text  # <class 'str'>
            # print(data)
            lines = data.split("\n")
            protein_id = ",".join([line.split("|")[1] for line in lines if line.startswith(">")])
            if protein_id :
                print(protein_id)
                data_noseq.loc[ind, "Protein_ID"] = protein_id
        else :
            print("Response Failed!")

    data_noseq.to_pickle(join(datasets_dir, annotate_dir, "EC_org_df_with_sequences_from_Uniprot.pkl"))

def protein_id_api(datasets_dir, annotate_dir) :
    EC_org_brenda = pd.read_pickle(join(datasets_dir, annotate_dir, "EC_org_df_with_sequences_from_brenda.pkl"))
    EC_org_brenda = EC_org_brenda[EC_org_brenda["Protein_ID"]!=""]
    EC_org_uniprot = pd.read_pickle(join(datasets_dir, annotate_dir, "EC_org_df_with_sequences_from_Uniprot.pkl"))
    # print(len(EC_org_brenda)+len(EC_org_uniprot))  # 19088
    EC_org_df = pd.concat([EC_org_brenda, EC_org_uniprot], ignore_index=True)
    EC_org_df = EC_org_df[EC_org_df["Protein_ID"]!=""]
    EC_org_df = EC_org_df[EC_org_df["Protein_ID"].str.split(',').apply(lambda x: len(x) == 1)]
    return EC_org_df

def get_protein_by_EC_org(df) :
    EC_org_single = df.copy()
    EC_org_single.loc[:, "EC_org"] = EC_org_single['EC'] + "&" + EC_org_single['ORGANISM']
    EC_org_protein_dict = dict(zip(EC_org_single["EC_org"], EC_org_single["Protein_ID"]))
    return EC_org_protein_dict

def annotate_protein(datasets_dir, annotate_dir) :
    EC_org_single = protein_id_api(datasets_dir, annotate_dir)
    EC_org_protein_dict = get_protein_by_EC_org(EC_org_single)
    print(len(EC_org_protein_dict))  # 3953
    categories = ["KCAT", "KM", "KCATKM"]
    for category in categories :
        print("This is for:", category.lower())
        data_df_kinetics = filter_smiles(datasets_dir, annotate_dir, typeFile=category)
        data_df_kinetics["Protein_ID"] = ""
        for index, row in data_df_kinetics.iterrows() :
            # print(row["UNIPROT"])
            if pd.notna(row["UNIPROT"]) :
                UNIPROT_list = row["UNIPROT"].split(",")
                # print(UNIPROT_list)
                if len(UNIPROT_list) == 1 :
                    data_df_kinetics.at[index, 'Protein_ID'] = row["UNIPROT"]
            else :
                EC = row["EC"]
                organism = row["ORGANISM"]
                EC_org = EC + "&" + organism 
                if EC_org in EC_org_protein_dict.keys() :
                    data_df_kinetics.at[index, 'Protein_ID'] = EC_org_protein_dict[EC_org]

        filtered_df = data_df_kinetics[data_df_kinetics["Protein_ID"]!=""]
        print("Entries with SMILES and Uniprot info:", len(filtered_df))
        # filtered_df = filtered_df[filtered_df["EnzymeType"]=="wildtype"]
        # print("Entries with wildtype enzymes:", len(filtered_df))
        filtered_df.reset_index(drop=True, inplace=True)

        # write output file
        filtered_df.to_csv(join(datasets_dir, annotate_dir, "data_df_%s_SMILES_UNIPROT.csv" % category), index=False)

def protein_id(datasets_dir, annotate_dir) :
    categories = ["KCAT", "KM", "KCATKM"]
    protein_list = list()
    for category in categories :
        print("This is for:", category.lower())
        data_df_kinetics = pd.read_csv(join(datasets_dir, annotate_dir, "data_df_%s_SMILES_UNIPROT.csv" % category))
        protein_data = data_df_kinetics["Protein_ID"].unique().tolist()
        print("Unique protein:", len(protein_data))
        protein_list += protein_data
    protein_list = list(set(protein_list))
    print("There exist %s unique protein identifiers." % len(protein_list))  # 13900

    with open(join(datasets_dir, annotate_dir, "proteinid.csv"), "w") as outfile :
        for protein in protein_list :
            outfile.write(protein +"\n")
        print("Genrerate a file containing all the protein identifiers.")


if __name__ == '__main__' :
    datasets_dir = "../complementaryData"
    annotate_dir = "data_annotation"
    brenda_query(datasets_dir, annotate_dir)
    uniprot_query(datasets_dir, annotate_dir)
    annotate_protein(datasets_dir, annotate_dir)
    protein_id(datasets_dir, annotate_dir)


