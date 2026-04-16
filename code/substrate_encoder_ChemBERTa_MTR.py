#!/usr/bin/python
# coding: utf-8
# Reference: https://arxiv.org/abs/2209.01712
# Date: 2024-09-27

import os
import torch
import pickle
import numpy as np
from os.path import join
from transformers import AutoTokenizer, AutoModel


SMILES_BERT = "DeepChem/ChemBERTa-77M-MTR"
SMILES_tokenizer = AutoTokenizer.from_pretrained(SMILES_BERT)
SMILES_bert = AutoModel.from_pretrained(SMILES_BERT)
SMILES_bert.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMILES_bert.to(device)

def get_last_layer_repr(smiles):
    tokens = SMILES_tokenizer(
        smiles,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = SMILES_bert(**tokens)
        hidden = outputs.last_hidden_state
    return hidden.cpu()

def create_smiles_embeddings_ChemBERTa_MTR(unique_smiles, out_path, smiles_embed_no) :
    if not os.path.exists(join(out_path, "SMILES", "ChemBERTa-77M-MTR")) :
        os.makedirs(join(out_path, "SMILES", "ChemBERTa-77M-MTR"), exist_ok=True)

    total_smiles = len(unique_smiles)
    parts = int(np.ceil(total_smiles/smiles_embed_no))

    for part in range(parts) :
        SMILES_reprs = dict()
        SMILES_list = unique_smiles[part*smiles_embed_no:(part+1)*smiles_embed_no]

        for k, smiles in enumerate(SMILES_list) :
            smiles_representation = get_last_layer_repr(smiles)
            SMILES_reprs[smiles] = smiles_representation

        with open(join(out_path, "SMILES", "ChemBERTa-77M-MTR", "SMILES_repr_"+str(part+1)+".pkl"), "wb") as outfile :
            pickle.dump(SMILES_reprs, outfile, protocol=pickle.HIGHEST_PROTOCOL)

