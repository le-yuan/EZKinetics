#!/usr/bin/python
# coding: utf-8
# Date: 2025-04-17
# Description: Generate UniMol embeddings from SMILES strings

import os
import pickle
import numpy as np
from os.path import join
from unimol_tools import UniMolRepr
import torch


unimol = UniMolRepr(data_type='molecule', remove_hs=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_unimol_repr(smiles):
    repr_dict = unimol.get_repr([smiles], return_atomic_reprs=False)
    return repr_dict["cls_repr"][0]

def create_smiles_embeddings_UniMol(unique_smiles, out_path, smiles_embed_no):
    save_dir = join(out_path, "SMILES", "UniMol")
    os.makedirs(save_dir, exist_ok=True)

    total_smiles = len(unique_smiles)
    parts = int(np.ceil(total_smiles / smiles_embed_no))

    for part in range(parts):
        SMILES_reprs = dict()
        SMILES_list = unique_smiles[part * smiles_embed_no:(part + 1) * smiles_embed_no]

        for k, smiles in enumerate(SMILES_list):
            try:
                smiles_representation = get_unimol_repr(smiles)
                SMILES_reprs[smiles] = smiles_representation
            except Exception as e:
                print(f"[Warning] Failed on {smiles}: {e}")
                continue

        save_path = join(save_dir, f"SMILES_repr_{part+1}.pkl")
        with open(save_path, "wb") as outfile:
            pickle.dump(SMILES_reprs, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(SMILES_reprs)} representations to {save_path}")
