import argparse
import os
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import pandas as pd
from collections import Counter
import itertools
import math
from typing import List, Tuple, Dict

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")  #20 standard amino acids. Honestly not sure if this is right, but let it be lol
all_dicodons = [''.join(p) for p in itertools.product(AA_ALPHABET, repeat=2)]


def findStartStop(seq: str, minlen_bp: int = 100) -> List[Tuple[int,int]]:
    """
    in essence : we go through all the letters, if its a START sequence, we add its index to the start sequences indexes
    if its a STOP sequence, we get the minimum in the start sequence indexes, and then clear the start positions and go again.
    """

    START = "ATG"
    STOPS = {"TAA","TAG","TGA"}

    seq = seq.upper()
    L = len(seq)
    orfs = []
    for frame in range(3):
        start_positions = []
        for codon_index in range(frame, L-2, 3):
            codon = seq[codon_index:codon_index+3]
            if codon in STOPS:
                if start_positions:
                    #get the earliest start (furthest away)
                    start_nt = min(start_positions)
                    end_nt = codon_index + 3
                    if (end_nt - start_nt) >= minlen_bp:
                        orfs.append((start_nt, end_nt))
                start_positions = []
            elif codon == START:
                start_positions.append(codon_index)
    return orfs

def findAllStartStopPairs(seq_record, minlen_bp=100) -> List[Dict]:
    """
    this basically gets all the ORF objects from normal and reverse sequences.
    returns start index, end index, strand (true if normal false if reversed, and the sequence itself)
    """
    seq = str(seq_record.seq).upper()
    orfs = []

    fs = findStartStop(seq, minlen_bp=minlen_bp)
    for s,e in fs:
        orf_seq = seq[s:e]
        orfs.append({'id': seq_record.id, 'start': s, 'end': e, 'strand': True, 'dna': orf_seq})


    rc = str(Seq(seq).reverse_complement())
    fs_rc = findStartStop(rc, minlen_bp=minlen_bp)

    L = len(seq)
    #gotta reverse the coordinates back to the original (unreversed) sequence coordinates
    for s,e in fs_rc:
        orig_start = L - e
        orig_end = L - s
        orf_seq = seq[orig_start:orig_end]  
        orfs.append({'id': seq_record.id, 'start': orig_start, 'end': orig_end, 'strand': False, 'dna': orf_seq})
    return orfs

def translate(dna_seq: str) -> str:
    prot = str(Seq(dna_seq).translate(to_stop=False))
    if prot.endswith("*"):
        prot = prot[:-1]
    return prot

# basically calculates the part each aminoacid takes in the protein. normalized to 0-1. Returns a map for all aminoacids
def aa_freqs(protein: str) -> Dict[str,float]:
    cnt = Counter(protein)
    total = sum(cnt[aa] for aa in AA_ALPHABET)
    freqs = {aa: (cnt.get(aa,0)/total if total>0 else 0.0) for aa in AA_ALPHABET}
    return freqs

#same as above, but in pairs. 
def dicodon_freqs(protein: str) -> Dict[str,float]:
    di_list = [protein[i:i+2] for i in range(len(protein)-1)]
    cnt = Counter(di_list)
    total = len(di_list)
    freqs = {di: (cnt.get(di,0)/total if total>0 else 0.0) for di in all_dicodons}
    return freqs


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.linalg.norm(vec1 - vec2))


def write_phylip(names: List[str], matrix: np.ndarray, outfile: str):
    n = len(names)
    with open(outfile, 'w', encoding="utf-8") as fh:
        fh.write(f"{n}\n")
        for i,name in enumerate(names):
            row = ' '.join(f"{matrix[i,j]:.6f}" for j in range(n))
            fh.write(f"{name} {row}\n")

def process_inputs(indir):
    records = []
    if indir:
        for fn in sorted(os.listdir(indir)):
            if not fn.endswith(".fasta"):
                continue
            path = os.path.join(indir, fn)
            for rec in SeqIO.parse(path, "fasta"):
                records.append(rec)
    return records

def main():
    os.makedirs("outdir", exist_ok=True)
    records = process_inputs(indir="data")
    if not records:
        print("error no input")
        return

    recordProteins = {}  
    all_orfs_info = [] 

    for rec in records:
        orfs = findAllStartStopPairs(rec, minlen_bp=100)
        prots = []
        for orf in orfs:
            dna = orf['dna']
            prot = translate(dna)
            if len(prot) == 0:
                continue
            prots.append(prot)
            all_orfs_info.append({
                'seq_id': rec.id,
                'start': orf['start'],
                'end': orf['end'],
                'strand': orf['strand'],
                'dna_len': len(dna),
                'aa_len': len(prot)
            })
        recordProteins[rec.id] = prots


    #join all ORF proteins
    merged_proteins = {}
    for rid, prots in recordProteins.items():
        merged_proteins[rid] = ''.join(prots) 

    #calculate aminoacid and dicodon frequency matrixes
    aa_freq_table = {}
    dicodon_freq_table = {}
    for rid, prot in merged_proteins.items():
        aa_freq_table[rid] = aa_freqs(prot)
        dicodon_freq_table[rid] = dicodon_freqs(prot)


    aa_df = pd.DataFrame.from_dict(aa_freq_table, orient='index').fillna(0.0).loc[sorted(merged_proteins.keys())]
    di_df = pd.DataFrame.from_dict(dicodon_freq_table, orient='index').fillna(0.0).loc[sorted(merged_proteins.keys())]

    #calculate distance matrixes
    ids = list(aa_df.index)
    n = len(ids)
    aa_mat = np.zeros((n,n))
    di_mat = np.zeros((n,n))

    for i in range(n):
        v1 = aa_df.iloc[i].values.astype(float)
        vd1 = di_df.iloc[i].values.astype(float)
        for j in range(n):
            v2 = aa_df.iloc[j].values.astype(float)
            vd2 = di_df.iloc[j].values.astype(float)
            aa_mat[i,j] = euclidean_distance(v1, v2)
            di_mat[i,j] = euclidean_distance(vd1, vd2)


    aa_phylip = os.path.join("outdir", "aa.phylip")
    di_phylip = os.path.join("outdir", "dicodon.phylip")
    write_phylip(ids, aa_mat, aa_phylip)
    write_phylip(ids, di_mat, di_phylip)

    aa_df.to_csv(os.path.join("outdir", "aa_freqs.csv"))
    di_df.to_csv(os.path.join("outdir", "dicodon_freqs.csv"))

    orf_df = pd.DataFrame(all_orfs_info)
    if not orf_df.empty:
        orf_df.to_csv(os.path.join("outdir", "orfs.csv"), index=False)

    print("Done, see outdir folder for results")

if __name__ == "__main__":
    main()