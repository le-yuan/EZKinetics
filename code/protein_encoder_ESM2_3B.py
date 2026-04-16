#!/usr/bin/python
# coding: utf-8
# Date: 2024-10-10

import os
import torch
import shutil
import pandas as pd
from os.path import join
from Bio import SeqIO
from esm import FastaBatchedDataset, pretrained


def create_seq_fasta(sequences, filename) :
    with open(filename, "w") as outfile :
        for k, seq in enumerate(sequences) :
            outfile.write(f">{k}\n{seq[:1022]}\n")

def merge_protein_emb_files(output_dir, out_path, fasta_file, prot_embed_no) :
    new_dict = {}
    version = 0
    fasta_sequences = SeqIO.parse(open(fasta_file),'fasta')

    for k, fasta in enumerate(fasta_sequences) :
        if k % prot_embed_no == 0 and k > 0 :
            torch.save(new_dict,join(out_path, "Protein", "ESM2_3B", "Protein_embeddings_V"+str(version+1)+".pt"))
            new_dict = {}
            version +=1

        name, sequence = fasta.id, str(fasta.seq)
        rep_dict = torch.load(join(output_dir, name +".pt"))
        new_dict[sequence] = rep_dict["representations"][33].numpy()

    torch.save(new_dict, join(out_path, "Protein", "ESM2_3B", "Protein_embeddings_V"+str(version+1)+".pt"))

    shutil.rmtree(output_dir)

def create_protein_embeddings_ESM2_3B(unique_sequences, out_path, prot_embed_no) :
    if not os.path.exists(join(out_path, "Protein", "ESM2_3B")) :
        os.makedirs(join(out_path, "Protein", "ESM2_3B"), exist_ok=True)
    fasta_file = join(out_path, "all_sequences.fasta")
    create_seq_fasta(unique_sequences, fasta_file)

    # https://huggingface.co/facebook/esm2_t36_3B_UR50D
    model, alphabet = pretrained.load_model_and_alphabet("esm2_t36_3B_UR50D")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Now using GPU")

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(4096, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches)

    output_dir = join(out_path, "Protein", "ESM2_3B", "temp")
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir, exist_ok=True)
        
    with torch.no_grad() :
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            # The model is trained on truncated sequences and passing longer ones in at
            # inference will cause an error. See https://github.com/facebookresearch/esm/issues/21
            toks = toks[:, :1022]

            out = model(toks, repr_layers=[33], return_contacts=False)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                output_file = join(output_dir, label + ".pt")
                
                result = {"label": label}
                result["representations"] = {
                    layer: t[i, 1 : len(strs[i]) + 1].clone()
                    for layer, t in representations.items()
                }
                
                torch.save(result, output_file)

    merge_protein_emb_files(output_dir, out_path, fasta_file, prot_embed_no)


