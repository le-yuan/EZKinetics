#!/usr/bin/python
# coding: utf-8
# Date: 2025-12-06

import os
import argparse
import pandas as pd
from os.path import join
# from protein_encoder_ESM3 import *
from protein_encoder_ESM2_650M import *
from protein_encoder_ESM2_3B import *
from protein_encoder_ProtT5 import *
from substrate_encoder_ChemBERTa_MTR import *
from substrate_encoder_ChemBERTa_MLM import *
from substrate_encoder_Uni_Mol import *


def get_arguments() :
    parser = argparse.ArgumentParser(description="Create protein embeddings and SMILES embeddings before training.")
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="This is the path where protein and SMILES embeddings are saved.",
    )

    parser.add_argument(
        "--prot_embed_no",
        type=int,
        default=2000,
        help="Number of protein sequences in one dictionary.",
    )

    parser.add_argument(
        "--smiles_embed_no",
        type=int,
        default=2000,
        help="Number of SMILES strings in one dictionary.",
    )
    args = parser.parse_args()
    return args

def main() :
    args = get_arguments()
    datasets_dir = "../data"
    sub_dir = "kinetics_data"
    typeFile = "KCAT"
    data_df_kinetics = pd.read_csv(join(datasets_dir, sub_dir, "data_%s.csv" % typeFile))

    # Find all the unique protein sequences and SMILES strings
    unique_sequences = data_df_kinetics["Sequence"].unique().tolist()
    unique_smiles = data_df_kinetics["SMILES"].unique().tolist()
    print("How many unique sequences:", len(unique_sequences))  # 12817
    print("How many unique substrate smiles:", len(unique_smiles))  # 5765

    if not os.path.exists(args.out_path) :
        os.makedirs(args.out_path, exist_ok=True)

    print("Start to create protein embeddings...")
    print("Method: ESM3")
    create_protein_embeddings_ESM3(unique_sequences, args.out_path, args.prot_embed_no)

    print("Start to create protein embeddings...")
    print("Method: ESM2_650M")
    create_protein_embeddings_ESM2_650M(unique_sequences, args.out_path, args.prot_embed_no)

    print("Start to create protein embeddings...")
    print("Method: ESM2_3B")
    create_protein_embeddings_ESM2_3B(unique_sequences, args.out_path, args.prot_embed_no)

    print("Start to create protein embeddings...")
    print("Method: ProtT5")
    create_protein_embeddings_ProtT5(unique_sequences, args.out_path, args.prot_embed_no)

    print("Start to create SMILES embeddings...")
    print("Method: ChemBERTa_MTR")
    create_smiles_embeddings_ChemBERTa_MTR(unique_smiles, args.out_path, args.smiles_embed_no)

    print("Start to create SMILES embeddings...")
    print("Method: ChemBERTa_MLM")
    create_smiles_embeddings_ChemBERTa_MLM(unique_smiles, args.out_path, args.smiles_embed_no)

    print("Start to create SMILES embeddings...")
    print("Method: Uni_Mol")
    create_smiles_embeddings_UniMol(unique_smiles, args.out_path, args.smiles_embed_no)

    print("Done!")


if __name__ == '__main__' :
    main()



