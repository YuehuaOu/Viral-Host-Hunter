import pandas as pd
import ast
import numpy as np
import h5py
from .embedding import get_embedding
from .encoder_only_dna import encoder


def compute_embedding(proteins, cds, ids, embedding_file):
    seq = []
    for i in range(len(proteins)):
        zj = ''
        for j in range(len(proteins[i]) - 1):
            zj += proteins[i][j] + ' '
        zj += proteins[i][-1]
        seq.append(zj)
    emb = get_embedding(seq)
    feature = encoder.features(cds)
    with h5py.File(embedding_file, 'w') as f:
        f.create_dataset('id', data=np.array(ids, dtype='S'), compression='gzip')
        f.create_dataset('protein_embedding', data=emb, compression='gzip')
        f.create_dataset('dna_embedding', data=feature, compression='gzip')


def take_embedding(embedding_file, ids):
    with h5py.File(embedding_file, 'r') as f:
        stored_ids = [sid.decode() for sid in f['id'][:]]
        idx_map = {sid: i for i, sid in enumerate(stored_ids)}
        indices = [idx_map[i] for i in ids]
        protein_embedding = f['protein_embedding'][indices]
        dna_embedding = f['dna_embedding'][indices]
    embedding = np.concatenate((protein_embedding, dna_embedding), axis=1)
    return embedding
