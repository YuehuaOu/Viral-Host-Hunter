import pandas as pd
import ast
import numpy as np
from embedding import get_embedding
from encoder_only_dna import encoder


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
    data = {'id': ids, 'protein_embedding': emb.tolist(), 'dna_embedding': feature.tolist()}
    df = pd.DataFrame(data)
    df.to_excel(embedding_file, index=False)


def take_embedding(embeddiing_file, ids):
    df = pd.read_excel(embeddiing_file)
    df.set_index("id", inplace=True)
    protein_embedding = np.array(df.loc[ids]["protein_embedding"].apply(ast.literal_eval))
    dna_embedding = np.array(df.loc[ids]["dna_embedding"].apply(ast.literal_eval))
    protein_embedding = np.vstack(protein_embedding)
    dna_embedding = np.vstack(dna_embedding)
    embedding = np.concatenate((protein_embedding, dna_embedding), axis=1)
    return embedding