import numpy as np
import pandas as pd
from Bio.SeqUtils import GC, CodonUsage


class Encoder:
    def __init__(self):
        pass

    def features(self, dna_list):
        dna_features = self.dna_features(dna_list)
        features = dna_features
        return features.values

    def dna_features(self, dna_list):
        A_freq, T_freq, C_freq, G_freq, GC_content = [], [], [], [], []
        codontable = {
            'ATA': [], 'ATC': [], 'ATT': [], 'ATG': [], 'ACA': [], 'ACC': [], 'ACG': [], 'ACT': [],
            'AAC': [], 'AAT': [], 'AAA': [], 'AAG': [], 'AGC': [], 'AGT': [], 'AGA': [], 'AGG': [],
            'CTA': [], 'CTC': [], 'CTG': [], 'CTT': [], 'CCA': [], 'CCC': [], 'CCG': [], 'CCT': [],
            'CAC': [], 'CAT': [], 'CAA': [], 'CAG': [], 'CGA': [], 'CGC': [], 'CGG': [], 'CGT': [],
            'GTA': [], 'GTC': [], 'GTG': [], 'GTT': [], 'GCA': [], 'GCC': [], 'GCG': [], 'GCT': [],
            'GAC': [], 'GAT': [], 'GAA': [], 'GAG': [], 'GGA': [], 'GGC': [], 'GGG': [], 'GGT': [],
            'TCA': [], 'TCC': [], 'TCG': [], 'TCT': [], 'TTC': [], 'TTT': [], 'TTA': [], 'TTG': [],
            'TAC': [], 'TAT': [], 'TAA': [], 'TAG': [], 'TGC': [], 'TGT': [], 'TGA': [], 'TGG': []
        }

        for item in dna_list:
            A_freq.append(item.count('A') / len(item))
            T_freq.append(item.count('T') / len(item))
            C_freq.append(item.count('C') / len(item))
            G_freq.append(item.count('G') / len(item))
            GC_content.append(GC(item))

            codons = [item[i: i+3] for i in range(0, len(item), 3)]
            l = []
            for key in codontable.keys():
                l.append(codons.count(key))
            l_norm = [float(i) / sum(l) for i in l]

            for j, key in enumerate(codontable.keys()):
                codontable[key].append(l_norm[j])

        synonym_codons = CodonUsage.SynonymousCodons
        codontable2 = {
            'ATA_b': [], 'ATC_b': [], 'ATT_b': [], 'ATG_b': [], 'ACA_b': [], 'ACC_b': [], 'ACG_b': [], 'ACT_b': [],
            'AAC_b': [], 'AAT_b': [], 'AAA_b': [], 'AAG_b': [], 'AGC_b': [], 'AGT_b': [], 'AGA_b': [], 'AGG_b': [],
            'CTA_b': [], 'CTC_b': [], 'CTG_b': [], 'CTT_b': [], 'CCA_b': [], 'CCC_b': [], 'CCG_b': [], 'CCT_b': [],
            'CAC_b': [], 'CAT_b': [], 'CAA_b': [], 'CAG_b': [], 'CGA_b': [], 'CGC_b': [], 'CGG_b': [], 'CGT_b': [],
            'GTA_b': [], 'GTC_b': [], 'GTG_b': [], 'GTT_b': [], 'GCA_b': [], 'GCC_b': [], 'GCG_b': [], 'GCT_b': [],
            'GAC_b': [], 'GAT_b': [], 'GAA_b': [], 'GAG_b': [], 'GGA_b': [], 'GGC_b': [], 'GGG_b': [], 'GGT_b': [],
            'TCA_b': [], 'TCC_b': [], 'TCG_b': [], 'TCT_b': [], 'TTC_b': [], 'TTT_b': [], 'TTA_b': [], 'TTG_b': [],
            'TAC_b': [], 'TAT_b': [], 'TAA_b': [], 'TAG_b': [], 'TGC_b': [], 'TGT_b': [], 'TGA_b': [], 'TGG_b': []
        }

        for item1 in dna_list:
            codons = [item1[l: l+3] for l in range(0, len(item1), 3)]
            codon_counts = []

            for key in codontable.keys():
                codon_counts.append(codons.count(key))

            for key_syn in synonym_codons.keys():
                total = 0
                for item2 in synonym_codons[key_syn]:
                    total += codons.count(item2)
                for j, key_table in enumerate(codontable.keys()):
                    if (key_table in synonym_codons[key_syn]) & (total != 0):
                        codon_counts[j] /= total

            for k, key_table in enumerate(codontable2.keys()):
                codontable2[key_table].append(codon_counts[k])

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
