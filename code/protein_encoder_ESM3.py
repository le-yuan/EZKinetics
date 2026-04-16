#!/usr/bin/python
# coding: utf-8
# Date: 2024-09-26

import os
import torch
import shutil
from os.path import join
from Bio import SeqIO
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL


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
            torch.save(new_dict,join(out_path, "Protein", "ESM3", "Protein_embeddings_V"+str(version+1)+".pt"))
            new_dict = {}
            version +=1

        name, sequence = fasta.id, str(fasta.seq)
        rep_dict = torch.load(join(output_dir, name +".pt"))
        new_dict[sequence] = rep_dict["per_residue_embedding"].numpy()

    torch.save(new_dict, join(out_path, "Protein", "ESM3", "Protein_embeddings_V"+str(version+1)+".pt"))

    shutil.rmtree(output_dir)

def create_protein_embeddings_ESM3(unique_sequences, out_path, prot_embed_no) :
    if not os.path.exists(join(out_path, "Protein", "ESM3")) :
        os.makedirs(join(out_path, "Protein", "ESM3"), exist_ok=True)
    fasta_file = join(out_path, "all_sequences.fasta")
    create_seq_fasta(unique_sequences, fasta_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    try:
        # Load the pre-trained model from Hugging Face
        client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    total_params = sum(param.numel() for param in client.parameters())
    print("Total parameters for ESM v3:", total_params)
    # 1401735748  1.4 billion parameters for ESM v3

    output_dir = join(out_path, "Protein", "ESM3", "temp")
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir, exist_ok=True)

    for seq_idx, seq in enumerate(unique_sequences):
        print(f"Processing sequence {seq_idx + 1} of {len(unique_sequences)}")
        protein = ESMProtein(sequence=seq[:1022])
        protein_tensor = client.encode(protein)
        # Perform forward pass and get per-residue embeddings
        output = client.forward_and_sample(
            protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
        )
        per_residue_embedding = output.per_residue_embedding.detach().cpu()
        result = {"label": str(seq_idx), "per_residue_embedding": per_residue_embedding}
        torch.save(result, join(output_dir, f"{seq_idx}.pt"))
        print(f"Embedding shape for sequence {seq_idx + 1}: {per_residue_embedding.shape}")
        # Example output
        # Processing sequence 1 of 23824
        # Embedding shape for sequence 1: torch.Size([330, 1536])
        # Processing sequence 2 of 23824
        # Embedding shape for sequence 2: torch.Size([377, 1536])

    # Merge and save the final embeddings in batches
    merge_protein_emb_files(output_dir, out_path, fasta_file, prot_embed_no)


