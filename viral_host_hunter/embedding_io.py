import os
import numpy as np
import h5py
from .embedding import get_embedding
from .encoder_only_dna import encoder

def _format_protein_sequences(proteins):
    seq = []
    for i in range(len(proteins)):
        zj = ''
        for j in range(len(proteins[i]) - 1):
            zj += proteins[i][j] + ' '
        zj += proteins[i][-1]
        seq.append(zj)
    return seq

def generate_protein_embedding(proteins, prott5_dir=None):
    seq = _format_protein_sequences(proteins)
    if prott5_dir is None:
        emb = get_embedding(seq)
    else:
        emb = get_embedding(seq, prott5_dir)
    return emb

def generate_dna_embedding(cds):
    return encoder.features(cds)

def save_hdf5(path, array):
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=array, compression='gzip')

def load_hdf5(path):
    with h5py.File(path, 'r') as f:
        return f['data'][:]

def ensure_embeddings(prot_path, dna_path, proteins, cds, prott5_dir=None):
    if not os.path.exists(prot_path):
        emb = generate_protein_embedding(proteins, prott5_dir)
        save_hdf5(prot_path, emb)
    if not os.path.exists(dna_path):
        feat = generate_dna_embedding(cds)
        save_hdf5(dna_path, feat)
