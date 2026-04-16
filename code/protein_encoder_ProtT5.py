#!/usr/bin/python
# coding: utf-8
# Reference: https://github.com/agemagician/ProtTrans/blob/master/Embedding/PyTorch/Advanced/ProtT5-XL-UniRef50.ipynb
# Paper: ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning
# Link: https://ieeexplore.ieee.org/document/9477085
# Date: 2024-10-28

import os
import re
import gc
import torch
import shutil
import pandas as pd
from os.path import join
from Bio import SeqIO
from transformers import T5EncoderModel, T5Tokenizer


def create_seq_fasta(sequences, filename) :
    with open(filename, "w") as outfile :
        for k, seq in enumerate(sequences) :
            outfile.write(f">{k}\n{seq[:1022]}\n")

def merge_protein_emb_files(output_dir, out_path, fasta_file, prot_embed_no):
    new_dict = {}
    version = 0
    fasta_sequences = SeqIO.parse(open(fasta_file), 'fasta')

    for k, fasta in enumerate(fasta_sequences):
        if k % prot_embed_no == 0 and k > 0:
            torch.save(new_dict, join(out_path, "Protein", "ProtT5", f"Protein_embeddings_V{version+1}.pt"))
            new_dict = {}
            version += 1

        name, sequence = fasta.id, str(fasta.seq)
        rep_dict = torch.load(join(output_dir, name + ".pt"))
        new_dict[sequence] = rep_dict["representations"]

    torch.save(new_dict, join(out_path, "Protein", "ProtT5", f"Protein_embeddings_V{version+1}.pt"))
    shutil.rmtree(output_dir)

def create_protein_embeddings_ProtT5(unique_sequences, out_path, prot_embed_no):
    if not os.path.exists(join(out_path, "Protein", "ProtT5")):
        os.makedirs(join(out_path, "Protein", "ProtT5"), exist_ok=True)
    fasta_file = join(out_path, "all_sequences.fasta")
    create_seq_fasta(unique_sequences, fasta_file)

    # Load the ProtT5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    gc.collect()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()

    output_dir = join(out_path, "Protein", "ProtT5", "temp")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    fasta_sequences = SeqIO.parse(open(fasta_file), 'fasta')

    for seq_idx, fasta in enumerate(fasta_sequences) :
        name, sequence = fasta.id, str(fasta.seq)
        sequence = re.sub(r"[UZOB]", "X", sequence)  # Replace non-standard amino acids
        print(f"Processing sequence {seq_idx + 1}")

        # Add spaces between amino acids
        spaced_sequence = [" ".join(sequence)]

        # Tokenize and encode the sequence
        ids = tokenizer.batch_encode_plus(spaced_sequence, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)

        # Remove padding (<pad>) and special tokens (</s>) that is added by ProtT5-XL-UniRef50 model
        embeddings = embedding.last_hidden_state.cpu().numpy()[0]
        seq_len = (attention_mask[0] == 1).sum().item()
        seq_embedding = embeddings[:seq_len - 1]
        print(f"Embedding shape for sequence {seq_idx + 1}: {seq_embedding.shape}")
        # Example output
        # Processing sequence 1
        # Embedding shape for sequence 1: (339, 1024)
        # Processing sequence 2
        # Embedding shape for sequence 2: (375, 1024)

        output_file = join(output_dir, name + ".pt")
        torch.save({"representations": seq_embedding}, output_file)
        torch.cuda.empty_cache()

    merge_protein_emb_files(output_dir, out_path, fasta_file, prot_embed_no)


