# import warnings
# warnings.filterwarnings("ignore", category=Warning)

import os
import gc
import argparse
import pickle

import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.utils.data import DataLoader, TensorDataset

from Bio import SeqIO

from . import utils
from .dataset import MyDataset
from .embedding_io import ensure_embeddings, load_hdf5
from .autoencoder import AutoEncoder
from .model import *
from .config import config
from .gut_prophages_info import info as gut_info
from .multi_taxonomic_levels_info import info as universal_info
from .get_phage_lineage import load_taxonomy, find_lineage
from .utils import int2str, autoencoder_encoder, feature_extract, test

 
def parse():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Viral-Host-Hunter prediction pipeline for viral host identification "
            "based on protein and DNA sequences."
        )
    )

    # ===== Required arguments =====
    parser.add_argument(
        '--protein', type=str, required=True,
        help='Path to the protein FASTA file to be predicted.'
    )
    parser.add_argument(
        '--dna', type=str, required=True,
        help='Path to the corresponding DNA FASTA file.'
    )
    parser.add_argument(
        '--seq_type', choices=['tail', 'lysin'], type=str, required=True,
        help='Protein type used for prediction: "tail" or "lysin".'
    )

    # ===== Model configuration =====
    parser.add_argument(
        '--phage_type', choices=['gut', 'environment'], type=str, default='gut',
        help='Phage source type: "gut" for intestinal phages or "environment" for environmental phages. (default: gut)'
    )
    parser.add_argument(
        '--level', choices=['all', 'family', 'genus', 'species'], type=str, default='all',
        help='Taxonomic prediction level: "all", "family", "genus", or "species". (default: all)'
    )
    parser.add_argument(
        '--model_dir', type=str,
        help='Directory containing the trained Viral-Host-Hunter models.'
    )

    # ===== Optional arguments =====
    parser.add_argument(
        '--embedding_dir', type=str, default='./embeddings',
        help=(
            'Directory to save/load precomputed embeddings (prot_embedding.h5, dna_embedding.h5). '
            'If embeddings already exist, they will be reused to speed up prediction. '
            '(default: ./embeddings)'
        )
    )
    parser.add_argument(
        '--output_dir', type=str, default='./output',
        help='Directory to save prediction results. (default: ./output)'
    )
    parser.add_argument(
        '--output_format', type=str, choices=['csv', 'tsv', 'xlsx', 'both'], default='csv',
        help='Output format for prediction results.'
    )
    parser.add_argument(
        '--prott5_dir', type=str, default=None,
        help=(
            'Path to a local ProtT5 model directory for offline embedding generation. '
            'Use this option if the system cannot download the model from the internet. '
            '(default: None)'
        )
    )
    parser.add_argument(
        '--lineage', action='store_true', default=False,
        help='If set, append lineage columns in the output.'
    )

    return parser.parse_args()


def get_thresholds():
    """Get threshold settings for different environments"""
    return {
        "gut": {
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
        },
        "environment": {
            "tail": {
                "family": {'95': 0.74, '84': 0.62, '69': 0.55, '-1': 0},
                "genus": {'95': 0.715, '84': 0.62, '69': 0.56, '-1': 0},
                "species": {'95': 0.75, '84': 0.62, '69': 0.535, '-1': 0}
            },
            "lysin": {
                "family": {'95': 0.695, '84': 0.605, '69': 0.545, '-1': 0},
                "genus": {'95': 0.785, '84': 0.595, '69': 0.535, '-1': 0},
                "species": {'95': 0.715, '84': 0.61, '69': 0.55, '-1': 0}
            }
        }
    }


def get_model_paths(model_dir,environment, type, level):
    """Get model paths based on the environment and parameters"""
    
    return {
        "model": f"{model_dir}/{environment}/{type}/{level}/model.pth",
        "autoencoder": f"{model_dir}/{environment}/{type}/{level}/autoencoder.pth", 
        "rf": f"{model_dir}/{environment}/{type}/{level}/rf.pth",
        "standard_scaler": f"{model_dir}/{environment}/{type}/{level}/standard_scaler.pkl"
    }



def main():
    # Set the random seed
    utils.set_seed(config.seed)
    
    # Parse command line arguments
    args = parse()
    
    # Extract arguments
    protein_file = args.protein
    dna_file = args.dna
    phage_type = args.phage_type
    seq_type = args.seq_type
    level = args.level
    model_dir = args.model_dir
    embedding_dir = args.embedding_dir
    output_dir = args.output_dir
    prott5_dir = args.prott5_dir
    use_lineage = args.lineage
    output_format = args.output_format
    
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Choose the corresponding info dictionary
    info = gut_info if phage_type == "gut" else universal_info
    
    # Get the threshold settings
    thresholds = get_thresholds()

    # Prepare lineage maps if requested
    lineage_maps = None
    if use_lineage:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        lineage_dir = os.path.join(base_dir, 'phage_lineage')
        excel_name = 'gut_phage_lineage.xlsx' if phage_type == 'gut' else 'environment_phage_lineage.xlsx'
        excel_path = os.path.join(lineage_dir, excel_name)
        lineage_maps = load_taxonomy(excel_path)
    
    print(f"Starting prediction - phage_type: {phage_type}, seq_type: {seq_type}, Level: {level}")
    
    # Read sequence data
    print("Reading sequence data...")
    test_cds = [str(rec.seq) for rec in SeqIO.parse(dna_file, 'fasta')]
    test_proteins = [str(rec.seq) for rec in SeqIO.parse(protein_file, 'fasta')]
    test_protein_ids = [str(rec.description) for rec in SeqIO.parse(protein_file, 'fasta')]
    test_dna_ids = [str(rec.description) for rec in SeqIO.parse(dna_file, 'fasta')]
    
    # Create dataset
    dummy_label = -1
    test_labels = [dummy_label] * len(test_proteins)
    test_dataset = MyDataset(test_cds, test_labels, config.k)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)
     
    
    # Generate embeddings
    print("Generating embeddings...")
    prot_h5 = os.path.join(embedding_dir, "prot_embedding.h5")
    dna_h5 = os.path.join(embedding_dir, "dna_embedding.h5")
    ensure_embeddings(prot_h5, dna_h5, test_proteins, test_cds, prott5_dir)

    # Load raw embedding arrays (shared across levels)
    test_prot_embed = load_hdf5(prot_h5)
    test_dna_embed = load_hdf5(dna_h5)
    concat_embedding = np.concatenate((test_prot_embed, test_dna_embed), axis=1)
    
    # Determine levels to run
    levels_to_run = ['family', 'genus', 'species'] if level == 'all' else [level]
    
    excel_path = os.path.join(output_dir, "predict_result.xlsx")
    writer = None
    if output_format in ['xlsx', 'both']:
        writer = pd.ExcelWriter(excel_path, engine='openpyxl')
    for lvl in levels_to_run:
            print(f"{'*' * 20}")
            print(f"Processing level: {lvl}")

            # Update class count and label mapping for this level
            config.num_class = max(info[seq_type][lvl].values()) + 1
            int2word_lvl = int2str(info[seq_type][lvl])
            if phage_type == "gut":
                int2word_lvl = {k: 'Mediterraneibacter gnavus' if v == 'Ruminococcus_gnavas' 
                               else 'Ruminococcus torques' if v == 'Ruminococcus_torques' 
                               else v for k, v in int2word_lvl.items()}
                int2word_lvl = {k: 'Bacillaceae' if v == 'Caryophanales' else v 
                               for k, v in int2word_lvl.items()}
            
            # Resolve model paths for this level
            model_paths_lvl = get_model_paths(model_dir, phage_type, seq_type, lvl)
            
            # Load scaler and transform embedding for this level
            print("Loading the standard scaler...")
            standard_scaler = utils.load_pickle(model_paths_lvl["standard_scaler"]) 
            test_embedding_lvl = standard_scaler.transform(concat_embedding)
            test_embedding_lvl = torch.from_numpy(test_embedding_lvl).float().to(config.device)
            
            # Load models
            print("Loading models...")
            DHH = DnaPathNetworks().to(config.device)
            DHH.load_state_dict(torch.load(model_paths_lvl["model"], weights_only=True, map_location=config.device))
            input_dim = 1280 * 4 + 1024 + 133
            hidden_dims = [4096, 2048, 1024]
            autoencoder = AutoEncoder(input_dim, hidden_dims).to(config.device)
            autoencoder.load_state_dict(torch.load(model_paths_lvl["autoencoder"], weights_only=True, map_location=config.device))
            rf = utils.load_pickle(model_paths_lvl["rf"]) 
            
            # Feature extraction and encoding
            print("Extracting features...")
            epoch_test_feature = feature_extract(DHH, test_dataloader, test_embedding_lvl)
            test_feature_dataset = TensorDataset(epoch_test_feature)
            test_feature_dataloader = DataLoader(test_feature_dataset, batch_size=10, shuffle=False)
            X_test = autoencoder_encoder(autoencoder, test_feature_dataloader)
            
            # Predictions
            print("Making predictions...")
            _, DHH_preds, DHH_prob = test(DHH, test_dataloader, test_embedding_lvl)
            rf_preds = rf.predict(X_test)
            rf_probs = rf.predict_proba(X_test)
            model_prob = 0.5 * np.array(DHH_prob) + 0.5 * rf_probs
            model_preds = np.argmax(model_prob, axis=1)
            preds = [int2word_lvl[item] for item in model_preds]
            prob = [model_prob[i][cls] for i, cls in enumerate(model_preds)]
            
            # Thresholded results for all confidence levels
            confidence_levels = ['-1', '95', '84', '69']
            threshold_results = {}
            threshold_lineage_results = {}
            for conf_level in confidence_levels:
                threshold = thresholds[phage_type][seq_type][lvl][conf_level]
                preds_th = []
                preds_th_lineage = []
                for i, p in enumerate(prob):
                    if p >= threshold:
                        preds_th.append(preds[i])
                    else:
                        preds_th.append("Unknown")
                # compute lineage columns if needed
                if use_lineage:
                    rank_key = {'family': 'family', 'genus': 'genes', 'species': 'species'}[lvl]
                    for name in preds_th:
                        if name == 'Unknown':
                            preds_th_lineage.append('Unknown')
                        else:
                            lineage_set = find_lineage(lineage_maps, name, rank=rank_key)
                            if not lineage_set:
                                preds_th_lineage.append('')
                            else:
                                preds_th_lineage.append(' | '.join(sorted(lineage_set)))
                    threshold_lineage_results[conf_level] = preds_th_lineage
                threshold_results[conf_level] = preds_th
            
            # Sheet DataFrame and write
            if use_lineage:
                results_data = {
                    'Protein_Desc': test_protein_ids,
                    'DNA_Desc': test_dna_ids,
                    'No_Threshold': threshold_results['-1'],
                    'No_Threshold_lineage': threshold_lineage_results.get('-1', [''] * len(test_protein_ids)),
                    'Confidence_69%': threshold_results['69'],
                    'Confidence_69%_lineage': threshold_lineage_results.get('69', [''] * len(test_protein_ids)),
                    'Confidence_84%': threshold_results['84'],
                    'Confidence_84%_lineage': threshold_lineage_results.get('84', [''] * len(test_protein_ids)),
                    'Confidence_95%': threshold_results['95'],
                    'Confidence_95%_lineage': threshold_lineage_results.get('95', [''] * len(test_protein_ids)),
                }
            else:
                results_data = {
                    'Protein_Desc': test_protein_ids,
                    'DNA_Desc': test_dna_ids,
                    'No_Threshold': threshold_results['-1'],
                    'Confidence_69%': threshold_results['69'],
                    'Confidence_84%': threshold_results['84'],
                    'Confidence_95%': threshold_results['95'],
                }
            results_df = pd.DataFrame(results_data)
            if writer is not None:
                results_df.to_excel(writer, sheet_name=lvl, index=False)
            if output_format in ['csv', 'both']:
                out_csv = os.path.join(output_dir, f"predict_result_{lvl}.csv")
                results_df.to_csv(out_csv, index=False)
            if output_format == 'tsv':
                out_tsv = os.path.join(output_dir, f"predict_result_{lvl}.tsv")
                results_df.to_csv(out_tsv, index=False, sep='\t')
            
            # Console summary per level
            # for conf_level in ['95', '84', '69']:
            #     high_conf_count = sum(1 for p in threshold_results[conf_level] if p != 'Unknown')
            #     print(f"High confidence predictions at {lvl} ({conf_level}%): {high_conf_count}")
            # print(f"No threshold predictions at {lvl}: {sum(1 for p in threshold_results['-1'] if p != 'Unknown')}")
            print("Done.")

            # Free GPU memory for this level to avoid accumulating VRAM usage
            try:
                del DHH
            except Exception:
                pass
            try:
                del autoencoder
            except Exception:
                pass
            try:
                del rf
            except Exception:
                pass
            # Intermediate tensors and arrays
            for _var in [
                'test_embedding_lvl', 'epoch_test_feature', 'test_feature_dataset',
                'test_feature_dataloader', 'X_test', 'DHH_preds', 'DHH_prob',
                'rf_preds', 'rf_probs', 'model_prob', 'model_preds', 'preds', 'prob'
            ]:
                try:
                    del locals()[_var]
                except Exception:
                    pass
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            gc.collect()

    if writer is not None:
        writer.close()
    print("\nPrediction complete!")
    if output_format in ['xlsx', 'both']:
        print(f"Saving Excel results to: {excel_path}")
    if output_format in ['csv', 'both']:
        print(f"Saving CSV results to: {output_dir}")
    if output_format == 'tsv':
        print(f"Saving TSV results to: {output_dir}")


if __name__ == "__main__":
    main()
