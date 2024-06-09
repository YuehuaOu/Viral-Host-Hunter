import torch
import numpy as np
from torch.utils.data import Dataset
import dna_encode


class MyDataset(Dataset):
    def __init__(self, cds, labels, k):
        X, Y = self.load_data(cds, labels, k)
        self.x = X
        self.y = Y

    def __getitem__(self, index):
        return index, self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def load_data(self, cds, labels, k):
        dna = cds
        dna_eb = map(lambda seq: self.encode_seq(seq, k), dna)
        emb = np.array(list(dna_eb))
        x = emb.reshape(-1, 6, 2 ** k, 2 ** k)
        X = torch.Tensor(x).float()
        label = labels
        label = np.array(label)
        Y = torch.from_numpy(label).long()
        return X, Y

    def encode_seq(self, seq, k):
        return dna_encode.matrix_encoding(seq, k)
