#!/usr/bin/python
# coding: utf-8
# Date: 2024-05-05

import json
from os.path import join
from urllib import request


# Example: https://rest.uniprot.org/uniprotkb/A0A1D8PIP5.fasta
def uniprot_sequence(id) :
    url = "https://rest.uniprot.org/uniprotkb/%s.fasta" % id
    IdSeq = dict()
    try :
        data = request.urlopen(url)
        respdata = data.read().decode("utf-8").strip()
        IdSeq[id] =  "".join(respdata.split("\n")[1:])
    except :
        # print(id, "can not find from uniprot!")
        IdSeq[id] = None

    return IdSeq[id]

def protein_id(datasets_dir, annotate_dir) :
    protein_ids = list()
    with open(join(datasets_dir, annotate_dir, "proteinid.csv"), 'r') as infile:
        lines = infile.readlines()
    for line in lines :
        data = line.strip()
        protein_ids.append(data)

    return protein_ids

def main() :
    datasets_dir = "../complementaryData"
    annotate_dir = "data_annotation"
    protein_ids = protein_id(datasets_dir, annotate_dir)
    print(len(protein_ids))  # 13900

    ProtID_seq = dict()
    ProtID_noseq = list()
    i = 0
    for UniprotID in protein_ids :
        i += 1
        print("This is ID", i)
        # print(UniprotID)
        sequence = uniprot_sequence(UniprotID)
        # print(sequence)
        if sequence :
            ProtID_seq[UniprotID] = sequence
        else :
            ProtID_noseq.append(UniprotID)

    print("Sequence can be obtained by Uniprot API:", len(ProtID_seq))        # 13682
    print("Sequence cannot be obtained by Uniprot API:", len(ProtID_noseq))   # 218

    with open(join(datasets_dir, annotate_dir, "uniprot_seq.json"), "w") as outfile1 :
        json.dump(ProtID_seq, outfile1, indent=4)
    with open(join(datasets_dir, annotate_dir, "uniprot_noseq.json"), "w") as outfile2 :
        json.dump(ProtID_noseq, outfile2, indent=4)

    print("Finished!")


if __name__ == '__main__' :
    main()


