import numpy as np


def _binary_transfer_AT(seq):
    seq = seq.replace("A", "0").replace("C", "1").replace("G", "1").replace("T", "0")
    seq = ''.join(filter(str.isdigit, seq))
    return seq


def _binary_transfer_AC(seq):
    seq = seq.replace("A", "0").replace("C", "0").replace("G", "1").replace("T", "1")
    seq = ''.join(filter(str.isdigit, seq))
    return seq


def _binary_transfer_loc(binary_seq, K):
    loc = []
    for i in range(0, len(binary_seq) - K + 1):
        loc.append(int(binary_seq[i:i + K], 2))
    return loc


def _loc_transfer_matrix(loc_list, dis, K):
    matrix = np.zeros((2 ** K, 2 ** K))
    for i in range(0, len(loc_list) - K - dis):
        matrix[loc_list[i]][loc_list[i + K + dis]] += 1
    return matrix


def _matrix_encoding(seq, K):
    length = len(seq)
    feature = []

    seq = seq.upper()
    binary_seq_1 = _binary_transfer_AT(seq)
    binary_seq_2 = _binary_transfer_AC(seq)
    loc_1 = _binary_transfer_loc(binary_seq_1, K)
    loc_2 = _binary_transfer_loc(binary_seq_2, K)

    for dis in range(3):
        feature.extend(_loc_transfer_matrix(loc_1, dis, K).flatten())
        feature.extend(_loc_transfer_matrix(loc_2, dis, K).flatten())

    return np.array(feature) / (length * 1.0) * 100


def matrix_encoding(seq, K):
    return _matrix_encoding(seq, K)
