import warnings

warnings.filterwarnings("ignore", category=Warning)

import argparse
from Bio import SeqIO
import pandas as pd
import openpyxl
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

from . import utils
from .utils import *
from .config import config
from .gut_prophages_info import info
from .embedding import get_embedding
from .encoder_only_dna import encoder
from .dataset import MyDataset
from .autoencoder import AutoEncoder
from .model import *

def parse():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Viral-Host-Hunter prediction pipeline for gut-associated phage host prediction "
            "based on protein and DNA sequence embeddings."
        )
    )

    parser.add_argument(
        '--protein_file', type=str, required=True,
        help='Path to the protein FASTA file for prediction.'
    )
    parser.add_argument(
        '--dna_file', type=str, required=True,
        help='Path to the corresponding DNA FASTA file for prediction.'
    )
    parser.add_argument(
        '--result_file', type=str, required=True,
        help='Path to save the prediction results.'
    )
    parser.add_argument(
        '--level', choices=['family', 'genus', 'species'], type=str, required=True,
        help='Taxonomic classification level for prediction: "family", "genus", or "species".'
    )
    parser.add_argument(
        '--type', choices=['tail', 'lysin'], type=str, required=True,
        help='Protein type used for prediction: "tail" or "lysin".'
    )
    parser.add_argument(
        '--precision', choices=['95', '84', '69', '-1'], type=str, required=True,
        help=(
            'Prediction confidence threshold. '
            '"95", "84", and "69" correspond to models with different confidence cutoffs; '
            '"-1" uses predictions without threshold filtering.'
        )
    )
    parser.add_argument(
        '--embedding_dir', type=str, default=None,
        help='Directory containing precomputed embedding files.'
    )
    parser.add_argument(
        '--model_dir', type=str,
        help="Directory containing trained gut phage prediction models. (default: ./models/gut_prophages/{type}/{level})"
    )
    parser.add_argument(
        '--prott5_dir', type=str, default=None,
        help=(
            'Path to a local ProtT5 model directory for offline embedding generation. '
            'Use this option if the system cannot download the model from the internet. '
            '(default: None)'
        )
    )

    return parser.parse_args()

def main():
    utils.set_seed(config.seed)
    args = parse()

    protein_file = args.protein_file
    dna_file = args.dna_file
    type = args.type
    level = args.level
    embedding_dir = args.embedding_dir
    result_file = args.result_file
    pre = args.precision
    model_dir = args.model_dir
    prott5_dir = args.prott5_dir

    if embedding_dir is None:
        embedding_dir = os.path.join('.', 'embedding', 'gut_prophages', type, level)
    os.makedirs(embedding_dir, exist_ok=True)
    print(f"Embedding directory: {embedding_dir}")
    if model_dir is None:
        model_dir = os.path.join('.', 'model', 'gut_prophages', type, level)
    print(f"Models directory: {model_dir}")
    print("Generating or loading embeddings ...")
    # generate embedding
    embedding_path1 = os.path.join(embedding_dir, "test_embedding.csv")
    embedding_path2 = os.path.join(embedding_dir, "test_dna_embed.csv")


    thresholds = {
        "tail": {
            "family": {'95': 0.72, '84': 0.65, '69': 0.59, '-1': 0},
            "genus": {'95': 0.705, '84': 0.625, '69': 0.57, '-1': 0},
            "species": {'95': 0.66, '84': 0.6, '69': 0.545, '-1': 0}
        },
        "lysin": {
            "family": {'95': 0.73, '84': 0.635, '69': 0.575, '-1': 0},
            "genus": {'95': 0.725, '84': 0.605, '69': 0.54, '-1': 0},
            "species": {'95': 0.675, '84': 0.58, '69': 0.52, '-1': 0}
        }
    }


    try:
        test_cds = [str(rec.seq) for rec in SeqIO.parse(dna_file, 'fasta')]
        test_proteins = [str(rec.seq) for rec in SeqIO.parse(protein_file, 'fasta')]
        test_hosts = [info[type][level][rec.description.split("#")[-1]] for rec in SeqIO.parse(protein_file, 'fasta')]
        hosts = [rec.description.split("#")[-1] for rec in SeqIO.parse(protein_file, 'fasta')]
    except KeyError as e:
        print(f"ERROR: KeyError occurred while processing test data: {e}")
        print(f"\nThe host organism '{e}' was not found in the predefined taxonomy mapping.")
        print(f"Please ensure your FASTA file headers follow the required format:")
        print(f"  - Header format: '>sequence_id_and_description #host_organism_name'")
        print(f"  - The host organism name (after #) must match exactly with the predefined taxonomy.")
        print(f"\nCurrent configuration:")
        print(f"  - Type: {type}")
        print(f"  - Level: {level}")
        print(f"  - Available hosts: {list(info[type][level].keys())}")
        print(f"\nPlease check your input files and ensure the host names match the expected format.")
        exit(1)
    test_labels = [max(info[type][level].values()) + 1 for i in range(len(test_hosts))]
    test_ids = [str(rec.description) for rec in SeqIO.parse(protein_file, 'fasta')]
    test_dataset = MyDataset(test_cds, test_labels, config.k)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # num_class
    config.num_class = max(info[type][level].values()) + 1

    int2word = int2str(info[type][level])
    int2word = {k: 'Mediterraneibacter gnavus' if v == 'Ruminococcus_gnavas' else 'Ruminococcus torques' if v == 'Ruminococcus_torques' else v for k, v in int2word.items()}
    int2word = {k: 'Bacillaceae' if v == 'Caryophanales' else v for k, v in int2word.items()}

    if not os.path.exists(embedding_path1):
        test_seq = []
        for i in range(len(test_proteins)):
            zj = ''
            for j in range(len(test_proteins[i]) - 1):
                zj += test_proteins[i][j] + ' '
            zj += test_proteins[i][-1]
            test_seq.append(zj)
        if prott5_dir is None:
            test_emb = get_embedding(test_seq)
        else:
            test_emb = get_embedding(test_seq, prott5_dir)
        np.savetxt(embedding_path1, test_emb, delimiter=',')

    if not os.path.exists(embedding_path2):
        test_feature = encoder.features(test_cds)
        np.savetxt(embedding_path2, test_feature, delimiter=',')

    # get embedding
    print("Loading standard scaler and models ...")
    standard_path = os.path.join(model_dir, 'standard_scaler.pkl')
    standard_scaler = StandardScaler()
    standard_scaler = utils.load_pickle(standard_path)

    test_embedding = pd.read_csv(os.path.join(embedding_dir, 'test_embedding.csv'), header=None)
    test_embedding = np.array(test_embedding)
    test_dna_embed = pd.read_csv(os.path.join(embedding_dir, 'test_dna_embed.csv'), header=None)
    test_dna_embed = np.array(test_dna_embed)
    test_embedding = np.concatenate((test_embedding, test_dna_embed), axis=1)
    test_embedding = standard_scaler.transform(test_embedding)
    test_embedding = torch.from_numpy(test_embedding).float().to(config.device)


    # load models
    DHH_path = os.path.join(model_dir, 'model.pth')
    DHH = DnaPathNetworks().to(config.device)
    DHH.load_state_dict(torch.load(DHH_path))

    input_dim = 1280 * 4 + 1024 + 133
    hidden_dims = [4096, 2048, 1024]
    autoencoder = AutoEncoder(input_dim, hidden_dims).to(config.device)
    autoencoder_path = os.path.join(model_dir, 'autoencoder.pth')
    autoencoder.load_state_dict(torch.load(autoencoder_path))

    rf_path = os.path.join(model_dir, 'rf.pth')
    rf = utils.load_pickle(rf_path)

    print("Extracting features and running predictions ...")
    # feature extract
    epoch_test_feature = feature_extract(DHH, test_dataloader, test_embedding)
    test_feature_dataset = TensorDataset(epoch_test_feature)
    test_feature_dataloader = DataLoader(test_feature_dataset, batch_size=10, shuffle=False)
    X_test = autoencoder_encoder(autoencoder, test_feature_dataloader)

    # predict
    # DHH
    _, DHH_preds, DHH_prob = test(DHH, test_dataloader, test_embedding)
    # RF
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)
    # merge
    model_prob = 0.5 * np.array(DHH_prob) + 0.5 * rf_probs
    model_preds = np.argmax(model_prob, axis=1)
    preds = []
    for item in model_preds:
        preds.append(int2word[item])

    prob = []
    for i, item in enumerate(model_preds):
        prob.append(model_prob[i][item])

    name_list = list(set(hosts))
    name = list(info[type][level].keys())
    check = all(item in name for item in name_list)

    threshold = thresholds[type][level][pre]
    preds_th = []
    for i, item in enumerate(prob):
        if item >= threshold:
            preds_th.append(preds[i])
    # save results using pandas
    df_results = pd.DataFrame({
        "ID": test_ids,
        "Host": hosts,
        "Pred": preds_th
    })
    df_results.to_csv(result_file, index=False)
    print(f"Saved prediction results to: {result_file}")

    if check:
        y_test = np.array(test_hosts)
        test_acc = accuracy_score(y_test, model_preds)
        precision = precision_score(y_test, model_preds, average='weighted')
        recall = recall_score(y_test, model_preds, average='weighted')
        f1 = f1_score(y_test, model_preds, average='weighted')
        print('Test ACC of classifier: %.4f' % (test_acc))
        print('Test precision of classifier: %.4f' % (precision))
        print('Test recall of classifier: %.4f' % (recall))
        print('Test f1 of classifier: %.4f' % (f1))
        print("======" * 20)
        test_precision = precision_score(y_test, model_preds, average=None)
        test_recall = recall_score(y_test, model_preds, average=None)
        name = list(info[type][level].keys())
        df = pd.DataFrame({'family host': name, 'Precision': test_precision, 'Recall': test_recall})
        print(df)

if __name__ == "__main__":
    main()
