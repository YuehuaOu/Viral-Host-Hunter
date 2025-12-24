# import warnings

# warnings.filterwarnings("ignore", category=Warning)

import argparse
from Bio import SeqIO
import pandas as pd
import h5py
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import lr_scheduler

from . import utils
from .utils import *
from .multi_taxonomic_levels_info import info
from .embedding_io import ensure_embeddings, load_hdf5
from .dataset import MyDataset
from .autoencoder import AutoEncoder
from .model import DnaPathNetworks
from .config import config

def parse():
        parser = argparse.ArgumentParser(
            description=(
                "Train the Viral-Host-Hunter model across multiple taxonomic levels.\n\n"
                "This script trains a multi-modal classifier using tail or lysin proteins "
                "and their corresponding DNA sequences to predict bacterial hosts "
                "at the family, genus, or species level."
            )
        )

        parser.add_argument(
            '--train_protein', type=str, required=True,
            help='Path to the training protein FASTA file.'
        )
        parser.add_argument(
            '--train_dna', type=str, required=True,
            help='Path to the corresponding training DNA FASTA file.'
        )
        parser.add_argument(
            '--val_protein', type=str, required=True,
            help='Path to the validation protein FASTA file.'
        )
        parser.add_argument(
            '--val_dna', type=str, required=True,
            help='Path to the corresponding validation DNA FASTA file.'
        )
        parser.add_argument(
            '--level', choices=['family', 'genus', 'species'], type=str, required=True,
            help='Taxonomic classification level of the prediction model: "family", "genus", or "species".'
        )
        parser.add_argument(
            '--type', choices=['tail', 'lysin'], type=str, required=True,
            help='Protein type used for training: "tail" or "lysin".'
        )
        parser.add_argument(
            '--output_dir', type=str,
            help="Directory to save trained model. (default: ./model/multi_taxonomic_levels/{type}/{level})"
        )
        parser.add_argument(
            '--embedding_dir', type=str, default=None,
            help=(
                'Directory to save/load precomputed embeddings (HDF5 files). '
                'If embeddings are already available, the model will reuse them to reduce computation time. '
                "(default: ./embeddings/multi_taxonomic_levels/{type}/{level})"
            )
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

    train_protein = args.train_protein
    train_dna = args.train_dna
    val_protein = args.val_protein
    val_dna = args.val_dna
    type = args.type
    level = args.level

    if args.embedding_dir is not None:
        embedding_file = args.embedding_dir
    else:
        embedding_file = os.path.join('.', 'embedding', 'multi_taxonomic_levels', type, level)

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join('.', 'model', 'multi_taxonomic_levels', type, level)
    prott5_dir = args.prott5_dir

    os.makedirs(embedding_file, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Embedding directory: {embedding_file}")
    print(f"Model output directory: {output_dir}")

    print("Loading data, this may take some time...")

    try:
        train_cds = [str(rec.seq) for rec in SeqIO.parse(train_dna, 'fasta')]
        train_proteins = [str(rec.seq) for rec in SeqIO.parse(train_protein, 'fasta')]
        train_labels = [info[type][level][rec.description.split("#")[-1]] for rec in SeqIO.parse(train_protein, 'fasta')]
    except KeyError as e:
        print(f"ERROR: KeyError occurred while processing training data: {e}")
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
    train_dataset = MyDataset(train_cds, train_labels, config.k)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    config.num_class = max(info[type][level].values()) + 1

    try:
        val_cds = [str(rec.seq) for rec in SeqIO.parse(val_dna, 'fasta')]
        val_proteins = [str(rec.seq) for rec in SeqIO.parse(val_protein, 'fasta')]
        val_labels = [info[type][level][rec.description.split("#")[-1]] for rec in SeqIO.parse(val_protein, 'fasta')]
    except KeyError as e:
        print(f"ERROR: KeyError occurred while processing validation data: {e}")
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
    val_dataset = MyDataset(val_cds, val_labels, config.k)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)


    print("Generating or loading embeddings ...")
    # generate embedding
    train_prot_h5 = os.path.join(embedding_file, "train_embedding.h5")
    train_dna_h5 = os.path.join(embedding_file, "train_dna_embed.h5")
    val_prot_h5 = os.path.join(embedding_file, "val_embedding.h5")
    val_dna_h5 = os.path.join(embedding_file, "val_dna_embed.h5")

    ensure_embeddings(train_prot_h5, train_dna_h5, train_proteins, train_cds, prott5_dir)

    ensure_embeddings(val_prot_h5, val_dna_h5, val_proteins, val_cds, prott5_dir)

    print("Fitting standard scaler on training embeddings ...")
    standard_scaler = StandardScaler()
    train_embedding = load_hdf5(train_prot_h5)
    train_dna_embed = load_hdf5(train_dna_h5)
    train_embedding = np.concatenate((train_embedding, train_dna_embed), axis=1)
    standard_scaler = standard_scaler.fit(train_embedding)
    utils.save_pickle(standard_scaler, os.path.join(output_dir, 'standard_scaler.pkl'))
    print(f"Saved standard scaler to: {os.path.join(output_dir, 'standard_scaler.pkl')}")

    val_embedding = load_hdf5(val_prot_h5)
    val_dna_embed = load_hdf5(val_dna_h5)
    val_embedding = np.concatenate((val_embedding, val_dna_embed), axis=1)

    print("Scaling embeddings ...")
    train_embedding = standard_scaler.transform(train_embedding)
    val_embedding = standard_scaler.transform(val_embedding)

    train_embedding = torch.from_numpy(train_embedding).float().to(config.device)
    val_embedding = torch.from_numpy(val_embedding).float().to(config.device)

    print("Initializing DHH ...")
    # initialize DHH
    DHH = DnaPathNetworks().to(config.device)
    optimizer = optim.Adam(DHH.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_function = torch.nn.CrossEntropyLoss().to(config.device)

    print("Training DHH ...")
    # train DHH
    best_acc = float('-inf')
    for epoch in range(1, config.epochs + 1):
        train_acc = utils.train(DHH, train_dataloader, train_embedding, loss_function, optimizer)
        scheduler.step()
        val_acc = utils.eval(DHH, val_dataloader, val_embedding)
        print('Epoch: %03d | Train ACC: %.4f | Val ACC: %.4f' % (epoch, train_acc, val_acc))
        if val_acc > best_acc:
            print('Val ACC improved, from %.4f to %.4f' % (best_acc, val_acc))
            best_acc = val_acc
            torch.save(DHH.state_dict(), os.path.join(output_dir, 'model.pth'))
            print(f"Saved best DHH model to: {os.path.join(output_dir, 'model.pth')}")
        else:
            print('Val ACC did not improve from %.4f' % (best_acc))


    print("Training autoencoder ...")
    # train autoencoder
    DHH = DnaPathNetworks().to(config.device)
    DHH.load_state_dict(torch.load(os.path.join(output_dir, 'model.pth'), weights_only=True))

    train_dataloader_neat = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)

    # feature extract
    epoch_train_feature = feature_extract(DHH, train_dataloader_neat, train_embedding)
    epoch_val_feature = feature_extract(DHH, val_dataloader, val_embedding)

    train_feature_dataset = TensorDataset(epoch_train_feature)
    train_feature_dataloader = DataLoader(train_feature_dataset, batch_size=config.batch_size, shuffle=True)
    val_feature_dataset = TensorDataset(epoch_val_feature)
    val_feature_dataloader = DataLoader(val_feature_dataset, batch_size=config.batch_size, shuffle=False)

    # initialize autoencoder
    input_dim = 1280 * 4 + 1024 + 133
    hidden_dims = [4096, 2048, 1024]
    autoencoder = AutoEncoder(input_dim, hidden_dims).to(config.device)
    loss_function = torch.nn.MSELoss().to(config.device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, verbose=True)

    # train
    best_val_loss = float('inf')
    for epoch in range(1, 150 + 1):
        train_loss = autoencoder_train(autoencoder, train_feature_dataloader, loss_function, optimizer)
        val_loss = autoencoder_eval(autoencoder, val_feature_dataloader, loss_function)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(autoencoder.state_dict(), os.path.join(output_dir, 'autoencoder.pth'))
    print(f"Saved best autoencoder to: {os.path.join(output_dir, 'autoencoder.pth')}")

    print("Training RandomForest ...")
    # train RF
    train_feature_dataset = TensorDataset(epoch_train_feature)
    train_feature_dataloader = DataLoader(train_feature_dataset, batch_size=config.batch_size, shuffle=False)

    autoencoder = AutoEncoder(input_dim, hidden_dims).to(config.device)
    autoencoder.load_state_dict(torch.load(os.path.join(output_dir, 'autoencoder.pth'), weights_only=True))

    X_train = autoencoder_encoder(autoencoder, train_feature_dataloader)
    X_val = autoencoder_encoder(autoencoder, val_feature_dataloader)
    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    # train
    best_acc = float('-inf')
    for min_samples_leaf in [1, 2, 3, 4]:
        for min_samples_split in [2, 3, 4]:
            for n_estimators in [50, 100, 150, 200]:
                clf = RandomForestClassifier(random_state=42, class_weight='balanced',
                                            max_features='sqrt',
                                            min_samples_leaf=min_samples_leaf,
                                            min_samples_split=min_samples_split,
                                            n_estimators=n_estimators,
                                            n_jobs=-1)
                clf = clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)
                val_acc = accuracy_score(y_val, y_pred)
                if val_acc > best_acc:
                    best_acc = val_acc
                    utils.save_pickle(clf, os.path.join(output_dir, 'rf.pth'))
    print(f"Saved best RandomForest to: {os.path.join(output_dir, 'rf.pth')}")

    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
