#!/usr/bin/python
# coding: utf-8
# Date: 2026-04-16

import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
from esm import pretrained
from transformers import AutoTokenizer, AutoModel


def load_protein_model_3B():
    model, alphabet = pretrained.load_model_and_alphabet("esm2_t36_3B_UR50D")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, alphabet, device

def load_protein_model_650M():
    model, alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, alphabet, device

def compute_protein_embeddings(sequences, model, alphabet, device, batch_size=4):
    embeddings = []
    batch_converter = alphabet.get_batch_converter()
    sequences = [seq[:1022] for seq in sequences]

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        labels = [str(j) for j in range(i, i + len(batch_seqs))]
        batch = list(zip(labels, batch_seqs))

        _, _, toks = batch_converter(batch)
        toks = toks.to(device)

        with torch.no_grad():
            out = model(toks, repr_layers=[33], return_contacts=False)

        reps = out["representations"][33]

        for j, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            emb = reps[j, 1:seq_len + 1].mean(dim=0).cpu().numpy()
            embeddings.append(emb)

    return embeddings

def load_smiles_model():
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

def compute_smiles_embeddings(smiles_list, tokenizer, model, device):
    embeddings = []
    for smi in tqdm(smiles_list, desc="Embedding SMILES"):
        tokens = tokenizer(
            smi,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = model(**tokens)
            last_layer = outputs.last_hidden_state
        pooled = last_layer.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(pooled)
    return embeddings

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="EZKinetics prediction")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    args = parser.parse_args()

    input_df = pd.read_csv(args.input)

    # Required columns
    required_columns = ["Sequence", "SMILES"]
    for col in required_columns:
        if col not in input_df.columns:
            raise ValueError(f"Missing required column: {col}")

    sequences = input_df["Sequence"].tolist()
    smiles = input_df["SMILES"].tolist()

    # Optional columns
    if "pH" in input_df.columns:
        ph_values = input_df["pH"].to_numpy()
    else:
        ph_values = np.full(len(sequences), 7.0)

    if "Temperature" in input_df.columns:
        temp_values = input_df["Temperature"].to_numpy()
    else:
        temp_values = np.full(len(sequences), 25.0)

    substrate_model = "ChemBERTa-77M-MTR"
    model_name = "XGBoost"
    training_dirs = ["KCAT_KM", "KCAT", "KM"]

    smi_tokenizer, smi_model, smi_device = load_smiles_model()
    smi_embeds = compute_smiles_embeddings(smiles, smi_tokenizer, smi_model, smi_device)
    smi_embeds = np.array(smi_embeds)

    for training_dir in training_dirs:

        if training_dir in ["KCAT_KM", "KM"]:
            print(f"Using ESM2_3B for {training_dir}...")
            prot_model, prot_alphabet, device = load_protein_model_3B()
            protein_model = "ESM2_3B"

        elif training_dir == "KCAT":
            print("Using ESM2_650M for KCAT...")
            prot_model, prot_alphabet, device = load_protein_model_650M()
            protein_model = "ESM2_650M"

        prot_embeds = compute_protein_embeddings(
            sequences,
            prot_model,
            prot_alphabet,
            device
        )
        prot_embeds = np.array(prot_embeds)

        X = np.concatenate(
            [prot_embeds, smi_embeds, ph_values[:, None], temp_values[:, None]],
            axis=1
        )

        model_base = join(
            "../model",
            training_dir,
            model_name,
            f"{protein_model}&{substrate_model}"
        )

        model_path = join(model_base, f"XGBoost_model_{training_dir}.pkl")
        scaler_path = join(model_base, f"scaler_{training_dir}.pkl")
        xgb_model = load_pickle(model_path)
        scaler = load_pickle(scaler_path)

        X_scaled = scaler.transform(X)
        y_pred = xgb_model.predict(X_scaled)
        y_real = np.round(10 ** y_pred, 3)

        if training_dir == "KCAT_KM":
            input_df["kcat/Km (1/mM*1/s)"] = y_real
        elif training_dir == "KCAT":
            input_df["kcat (1/s)"] = y_real
        elif training_dir == "KM":
            input_df["Km (mM)"] = y_real

        del prot_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    input_df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()


