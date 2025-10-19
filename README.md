# Decrypting Viral Dark Matter with Predictive Framework and Therapeutic Implications

## 1. Introduction

This repository contains the source data and code for the paper **"Decrypting Viral Dark Matter with Predictive Framework and Therapeutic Implications"**.

The Viral-Host-Hunter (VHH) framework predicts bacterial hosts for phages based on protein and DNA sequences using multi-modal deep learning models.

## 2. Installation

### 2.1 Clone the Repository

```bash
git clone git@github.com:YuehuaOu/Viral-Host-Hunter.git
cd Viral-Host-Hunter
```

### 2.2 Environment Setup

We tested our code with **PyTorch 2.4.0 + CUDA 11.8**. It is recommended to replicate this environment for reproducibility:

```bash
# 1. Create a Python 3.8 environment
conda create -n VHH python=3.8

# 2. Activate environment
conda activate VHH

# 3. Install PyTorch compatible with CUDA 11.8
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Install 
conda install -c wlhuang viral-host-hunter
```

> **Notes**:
>
> - If your system has a different CUDA version, refer to [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) to select the correct installation.
> - Common error:
>
>   ```
>   RuntimeError: CUDA error: no kernel image is available for execution on the device
>   ```
>
>   This occurs when PyTorch and CUDA are incompatible. Install the appropriate PyTorch version for your GPU.

### 2.3 Pretrained Models

Download pretrained models from our [model repository](https://zenodo.org/records/17340381):

```bash
wget https://zenodo.org/records/17340381/files/models.zip
unzip models.zip
```

When using the `vhh-predict` command, you need to use the `--model_dir` parameter to point to the downloaded models directory.

Additionally, VHH requires the pretrained **ProtT5-XL-UniRef50** model, but don't worry:

- When you use our program, the model will automatically download if your machine has internet access.
- If running offline, download the model manually from [Rostlab/prot_t5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) and use the `--prott5_dir` parameter to point to the local path.

### 2.4 Test Example

We provide example data and example scripts in the example directory for quick verification of correct installation, but you need to add the `--model_dir` parameter to the command:

```bash
vhh-predict \
--protein ./example/gut/tail/protein.fasta \
--dna ./example/gut/tail/dna.fasta \
--phage_type gut --seq_type tail \
--embedding_dir ./example/embedding \
--output_dir ./example/output \
--model_dir <path_to_models>
```

For detailed command parameter descriptions, see below.

## 3 Usage

### 3.1 Basic Usage

Use the `vhh-predict` command to run predictions with our model. See `example/run_example.sh` for a usage example.

Run `vhh-predict` to view the command-line help message.

```
$ vhh-predict -h
usage: vhh-predict [-h] --protein PROTEIN --dna DNA --seq_type {tail,lysin} [--phage_type {gut,environment}]
                   [--level {all,family,genus,species}] [--model_dir MODEL_DIR] [--embedding_dir EMBEDDING_DIR]
                   [--output_dir OUTPUT_DIR] [--prott5_dir PROTT5_DIR] [--lineage]

Run the Viral-Host-Hunter prediction pipeline for viral host identification based on protein and DNA sequences.

optional arguments:
  -h, --help            show this help message and exit
  --protein PROTEIN     Path to the protein FASTA file to be predicted.
  --dna DNA             Path to the corresponding DNA FASTA file.
  --seq_type {tail,lysin}
                        Protein type used for prediction: "tail" or "lysin".
  --phage_type {gut,environment}
                        Phage source type: "gut" for intestinal phages or "environment" for environmental phages. (default:
                        gut)
  --level {all,family,genus,species}
                        Taxonomic prediction level: "all", "family", "genus", or "species". (default: all)
  --model_dir MODEL_DIR
                        Directory containing the trained Viral-Host-Hunter models.
  --embedding_dir EMBEDDING_DIR
                        Directory to save/load precomputed embeddings (refer to prot_embedding.csv, dna_embedding.csv). If
                        embeddings already exist, they will be reused to speed up prediction. (default: ./embeddings)
  --output_dir OUTPUT_DIR
                        Directory to save prediction results. (default: ./output)
  --prott5_dir PROTT5_DIR
                        Path to a local ProtT5 model directory for offline embedding generation. Use this option if the
                        system cannot download the model from the internet. (default: None)
  --lineage             If set, append lineage columns in the output
```

The embedding generation using ProtT5-XL-UniRef50 requires GPU. To ensure optimal speed and accuracy, we strongly recommend using a GPU environment, although we have tested that CPU-only execution is possible on the example data. We utilized an NVIDIA GeForce RTX 3090 with 24GiB VRAM to embed peptide or protein sequences into 1024-dimensional vectors.

### 3.2 Output Description

You will find the prediction results in `$OUTPUT_DIR/predict_result.xlsx`. This file contains three sheets corresponding to family, genus, and species level predictions, respectively. Each sheet follows the format shown in the table below:

- Columns 1-2 contain the descriptions of the input protein sequence and DNA sequence, respectively;
- Columns 4-7 show the predicted host results for no threshold, 69% threshold, 84% threshold, and 95% threshold, respectively.

| Protein_Desc    | DNA_Desc    | No_Threshold     | Confidence_69%   | Confidence_84%   | Confidence_95%   |
| --------------- | ----------- | ---------------- | ---------------- | ---------------- | ---------------- |
| tail_1 #protein | tail_1 #dna | Eubacteriaceae   | Eubacteriaceae   | Unknown          | Unknown          |
| tail_2 #protein | tail_2 #dna | Eubacteriaceae   | Eubacteriaceae   | Eubacteriaceae   | Eubacteriaceae   |
| tail_3 #protein | tail_3 #dna | Xanthomonadaceae | Xanthomonadaceae | Xanthomonadaceae | Xanthomonadaceae |
| tail_4 #protein | tail_4 #dna | Xanthomonadaceae | Xanthomonadaceae | Xanthomonadaceae | Xanthomonadaceae |
| tail_5 #protein | tail_5 #dna | Bacteroidaceae   | Bacteroidaceae   | Unknown          | Unknown          |

If you specify the `--lineage` parameter, the results will also include the host lineage for the predicted results.

## 4 Training

### 4.1 Data Preparation

To retrain our models using the data from our paper, download our dataset using the following commands. The data has already been split into train, validation, and test sets according to the methods described in the paper:

```bash
wget https://zenodo.org/records/17340915/files/data.zip
unzip data.zip
```

### 4.2 Training

Use the `vhh-train-gut` and `vhh-train-multi` commands to train models for the gut prophages dataset and multi-taxonomic levels dataset, respectively.

Example usage of `vhh-train-gut`:

```bash
vhh-train-gut \
--train_protein <path_to_data>/gut_prophages/lysin/species/train_protein.fasta \
--train_dna <path_to_data>/gut_prophages/lysin/species/train_dna.fasta \
--val_protein <path_to_data>/gut_prophages/lysin/species/val_protein.fasta \
--val_dna <path_to_data>/gut_prophages/lysin/species/val_dna.fasta \
--type lysin \
--level species
```

Use `--type` and `--level` to train models for different phage types and taxonomic classification levels. Run `vhh-train-gut -h` to view the command-line help message.


Similarly, for the multi-taxonomic levels dataset, you can run the `vhh-train-multi` command to train. For example:

```bash
vhh-train-multi \
--train_protein ./data/multi_taxonomic_levels/tail/family/train_protein.fasta \
--train_dna ./data/multi_taxonomic_levels/tail/family/train_dna.fasta \
--val_protein ./data/multi_taxonomic_levels/tail/family/val_protein.fasta \
--val_dna ./data/multi_taxonomic_levels/tail/family/val_dna.fasta \
--type tail \
--level family
```

This script has the same parameters as the one above, which you can refer to. Run `vhh-train-multi -h` to view the command-line help message.

### 4.3 Prediction

Use the `vhh-test-gut` and `vhh-test-multi` commands to make predictions using the models trained in the previous step. For the convenience of reproducing results, these two programs require test data with labels to calculate evaluation metrics. (For test data without labels, use the program in section 3.1 to obtain prediction results)

For the gut prophages dataset, you can use the `vhh-predict-gut` command to implement VHH evaluation on this dataset.

```bash
vhh-predict-gut --protein_file protein.fasta \
 --dna_file cds.fasta \
 --type tail \
 --level family \
 --precision -1 \
 --embedding_name test \
 --result_file results.csv
```

You can also set the confidence threshold of VHH through the `--precision` parameter (optional thresholds are 95%, 84%, 69%) to obtain higher confidence results.

If you used the `--output_dir` parameter to specify the model save path during training, you need to use `--model_dir` to specify that path during prediction.

Run `vhh-predict-gut -h` to view the command-line help message.

```
$ vhh-predict-gut -h
usage: vhh-predict-gut [-h] --protein_file PROTEIN_FILE --dna_file DNA_FILE --result_file RESULT_FILE --level {family,genus,species} --type {tail,lysin} --precision
                       {95,84,69,-1} [--embedding_dir EMBEDDING_DIR] [--model_dir MODEL_DIR] [--prott5_dir PROTT5_DIR]

Run the Viral-Host-Hunter prediction pipeline for gut-associated phage host prediction based on protein and DNA sequence embeddings.

optional arguments:
  -h, --help            show this help message and exit
  --protein_file PROTEIN_FILE
                        Path to the protein FASTA file for prediction.
  --dna_file DNA_FILE   Path to the corresponding DNA FASTA file for prediction.
  --result_file RESULT_FILE
                        Path to save the prediction results.
  --level {family,genus,species}
                        Taxonomic classification level for prediction: "family", "genus", or "species".
  --type {tail,lysin}   Protein type used for prediction: "tail" or "lysin".
  --precision {95,84,69,-1}
                        Prediction confidence threshold. "95", "84", and "69" correspond to models with different confidence cutoffs; "-1" uses predictions without
                        threshold filtering.
  --embedding_dir EMBEDDING_DIR
                        Directory containing precomputed embedding files.
  --model_dir MODEL_DIR
                        Directory containing trained gut phage prediction models. (default: ./models/gut_prophages/{type}/{level})
  --prott5_dir PROTT5_DIR
                        Path to a local ProtT5 model directory for offline embedding generation. Use this option if the system cannot download the model from the
                        internet. (default: None)
```

For the multi-taxonomic levels dataset, you can use the `vhh-predict-multi` command to implement VHH evaluation on this dataset.

For example:

```bash
vhh-predict-multi \
--protein_file ./data/multi_taxonomic_levels/tail/family/test_protein.fasta \
--dna_file ./data/multi_taxonomic_levels/tail/family/test_dna.fasta \
--type tail \
--level family \
--precision -1 \
--embedding_name predict \
--result_file results.csv
```

This script has the same parameters as the one above, which you can refer to. Run `vhh-predict-multi -h` to view the command-line help message.

### 4.4 Training with Your Own Data

**Preparing your own dataset**

If you want to retrain our model using a new dataset, you need to prepare both protein and dna sequence FASTA files in the following format:

- Add host information in the format of `#<host>` at the end of each FASTA definition line for the program to extract labels. For example:

```fasta
>GCF_944325205_gene_1582 #Desulfovibrionaceae
MADFDLAYAPVSKWEGGWTHDSGDKGGETFRGCARNFFPNEPIWPVIDREKSHPSYKQGK
AAFSAHLMGIPSLTGCVKGWYRKEWWDKLGLERFDQIVADELFEQAVNLGKAGMGRYLQR
LCNAFNWRKDGSADGARLFDDLQTDGVVGPKTLSALSIVLSRNDARRIVHLMNCMQGAHY
```

- Each protein sequence in the protein FASTA file must correspond one-to-one with a DNA sequence in the DNA FASTA file, sharing the same sequence identifier (`#<host>` tag). For example:

```
Protein FASTA file:
>GCF_944325205_gene_1582 #Desulfovibrionaceae
MADFDLAYAPVSKWEGGWTHDSGDKGGETFRGCARNFFPNEPIWPVIDREKSHPSYKQGK
AAFSAHLMGIPSLTGCVKGWYRKEWWDKLGLERFDQIVADELFEQAVNLGKAGMGRYLQR
LCNAFNWRKDGSADGARLFDDLQTDGVVGPKTLSALSIVLSRNDARRIVHLMNCMQGAHY
```

```
DNA FASTA file:
>GCF_944325205_gene_1582 #Desulfovibrionaceae
ATGGCTGATTTTGATCTGGCGTATGCTCCAGTTTCCAAGTGGGAAGGAGGATGGACCCAT
GATTCAGGCGATAAAGGCGGTGGCGAAGTTCCGCGGTGCGGCCCGGAATTTTTTCCGAAT
GAACCCATCTGGCCGGTCATTGACCGTGAAAAGAGCCACCCGTCATACAAACAGGGCAAG
```

**Modify the code and provide label information**

Once your new dataset has been prepared, follow the steps below to update the code and label information for retraining and prediction.

- Prepare label information for the new dataset： You need to provide the label information corresponding to your new dataset. As a reference, check the file `multi_taxonomic_levels_info.py`, which contains the original label definitions. Create a new file (for example, `new_data_info.py`) following the same structure, and replace the existing labels with those from your new dataset.

- Modify the training code: To train the model with your new dataset, modify the training script accordingly.  You can refer to `train_multi_taxonomic_levels.py` as an example. Specifically, import your new label information file by replacing:

```
from .multi_taxonomic_levels_info import info
```

with:

```
from .new_data_info import info
```

After this modification, you can proceed to train the model using your new dataset.

- Modify the prediction code: When performing prediction, make similar adjustments to the prediction script as done during training. Refer to `predict_multi_taxonomic_levels.py`, and replace the label import in the same way. Note that during prediction, the input data should follow the same structure as in training — it must include both protein and DNA sequence FASTA files, with each protein sequence corresponding one-to-one to a DNA sequence.

## Contact Information

1. Yuehua Ou, ouyuehua2022@email.szu.edu.cn
2. Zihao Lin, 2410103047@mails.szu.edu.cn

## Copyright Information / License

Please see the "LICENSE.txt" file for the copyright information.

