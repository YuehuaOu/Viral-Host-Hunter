import numpy as np
import pandas as pd
from Bio.SeqUtils import gc_fraction

synonymous_codons = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'C': ['TGT', 'TGC'],
    'D': ['GAT', 'GAC'],
    'E': ['GAA', 'GAG'],
    'F': ['TTT', 'TTC'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    'H': ['CAT', 'CAC'],
    'I': ['ATT', 'ATC', 'ATA'],
    'K': ['AAA', 'AAG'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'M': ['ATG'],
    'N': ['AAT', 'AAC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'Q': ['CAA', 'CAG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    'W': ['TGG'],
    'Y': ['TAT', 'TAC'],
    '*': ['TAA', 'TAG', 'TGA'],
}

class Encoder:
    def __init__(self):
        pass

    def features(self, dna_list):
        dna_features = self.dna_features(dna_list)
        return dna_features.values

    def dna_features(self, dna_list):
        A_freq, T_freq, C_freq, G_freq, GC_content = [], [], [], [], []
        codontable = {codon: [] for codon in [
            'ATA','ATC','ATT','ATG','ACA','ACC','ACG','ACT','AAC','AAT','AAA','AAG',
            'AGC','AGT','AGA','AGG','CTA','CTC','CTG','CTT','CCA','CCC','CCG','CCT',
            'CAC','CAT','CAA','CAG','CGA','CGC','CGG','CGT','GTA','GTC','GTG','GTT',
            'GCA','GCC','GCG','GCT','GAC','GAT','GAA','GAG','GGA','GGC','GGG','GGT',
            'TCA','TCC','TCG','TCT','TTC','TTT','TTA','TTG','TAC','TAT','TAA','TAG',
            'TGC','TGT','TGA','TGG']}

        try:
            for seq in dna_list:
                A_freq.append(seq.count('A') / len(seq))
                T_freq.append(seq.count('T') / len(seq))
                C_freq.append(seq.count('C') / len(seq))
                G_freq.append(seq.count('G') / len(seq))
                GC_content.append(gc_fraction(seq) * 100)

                codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
                codon_counts = [codons.count(c) for c in codontable.keys()]
                total = sum(codon_counts)
                if total == 0:
                    total = 1
                codon_norm = [c/total for c in codon_counts]

                for j, key in enumerate(codontable.keys()):
                    codontable[key].append(codon_norm[j])
        except Exception as e:
            print(f"[ERROR] Failed computing base freq for seq {seq}: {e}")
            exit(1)

        # codon bias features
        codontable2 = {k+'_b': [] for k in codontable.keys()}

        for seq in dna_list:
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            codon_counts = [codons.count(c) for c in codontable.keys()]

            for key_aa, codon_list in synonymous_codons.items():
                total = sum([codons.count(c) for c in codon_list])
                if total > 0:
                    for i, codon in enumerate(codontable.keys()):
                        if codon in codon_list:
                            codon_counts[i] /= total

            for k, key in enumerate(codontable2.keys()):
                codontable2[key].append(codon_counts[k])

        features_codonbias = pd.DataFrame.from_dict(codontable2)
        features_dna = pd.DataFrame.from_dict(codontable)
        features_dna['A_freq'] = np.asarray(A_freq)
        features_dna['T_freq'] = np.asarray(T_freq)
        features_dna['C_freq'] = np.asarray(C_freq)
        features_dna['G_freq'] = np.asarray(G_freq)
        features_dna['GC'] = np.asarray(GC_content)

        features = pd.concat([features_dna, features_codonbias], axis=1)
        return features

encoder = Encoder()
