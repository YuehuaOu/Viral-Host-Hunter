# Decrypting Viral Dark Matter with Predictive Framework and Therapeutic Implications

## 1. Introduction

This repository contains the source data and code for the paper **"Decrypting Viral Dark Matter with Predictive Framework and Therapeutic Implications"**.

The Viral-Host-Hunter (VHH) framework predicts bacterial hosts for phages based on protein and DNA sequences using multi-modal deep learning models.

------

## 2. Installation

### 2.1 Clone the Repository

```bash
git clone git@github.com:YuehuaOu/Viral-Host-Hunter.git
cd Viral-Host-Hunter
```

------

### 2.2 Environment Setup

We tested our code in **PyTorch 2.4.0 + CUDA 11.8**. It is recommended to replicate the environment for reproducibility:

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
>
> - Common error:
>
>   ```
>   RuntimeError: CUDA error: no kernel image is available for execution on the device
>   ```
>
>   This occurs when PyTorch and CUDA are incompatible. Install the appropriate PyTorch version for your GPU.

------

### 2.3 Pretrained Models

Download pretrained models from our [model repository](link) and place them under `Viral-Host-Hunter/models/`:

```
Viral-Host-Hunter/
├── models/            # Pretrained model weights
├── example/           # Example data and scripts
├── viral_host_hunter         # Main code
├── README.md
└──...
```

Alternatively, you can specify the model directory path using the `--model_dir` flag in the `vhh-predict` command (see below).

------

### 2.4 ProtT5-XL-UniRef50

VHH requires the pretrained **ProtT5-XL-UniRef50** model.

- The model will automatically download if your machine has internet access.
- If running offline, download the model manually from [Rostlab/prot_t5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) and use the `--prott5_dir` parameter to point to the local path.

```bash
bash example/run_example.sh
```

------

### 2.5 Test Example

Run the example script to check installation:

```bash
bash example/run_example.sh
```

You may need to add `--model_dir` or `--prott5_dir` parameters if using custom paths.

------

## 3 Usage

### 3.1 Basic Usage

Use the `vhh-predict` command to run predictions with our model. See `example/run_example.sh` for a usage example.

Run `vhh-predict` to view the command-line help message.

```
$ vhh-predict -h
usage: predict.py [-h] --protein PROTEIN --dna DNA --seq_type {tail,lysin} [--phage_type {gut,environment}] [--level {all,family,genus,species}] --model_dir
                  MODEL_DIR [--embedding_dir EMBEDDING_DIR] [--output_dir OUTPUT_DIR] [--prott5_dir PROTT5_DIR]

Run the Viral-Host-Hunter prediction pipeline for viral host identification based on protein and DNA sequences.

optional arguments:
  -h, --help            show this help message and exit
  --protein PROTEIN     Path to the protein FASTA file to be predicted.
  --dna DNA             Path to the corresponding DNA FASTA file.
  --seq_type {tail,lysin}
                        Protein type used for prediction: "tail" or "lysin".
  --phage_type {gut,environment}
                        Phage source type: "gut" for intestinal phages or "environment" for environmental phages. (default: gut)
  --level {all,family,genus,species}
                        Taxonomic prediction level: "all", "family", "genus", or "species". (default: all)
  --model_dir MODEL_DIR
                        Directory containing the trained Viral-Host-Hunter models.
  --embedding_dir EMBEDDING_DIR
                        Directory to save/load precomputed embeddings (refer to prot_embedding.csv, dna_embedding.csv). If embeddings already exist, they will be
                        reused to speed up prediction. (default: ./embeddings)
  --output_dir OUTPUT_DIR
                        Directory to save prediction results. (default: ./output)
  --prott5_dir PROTT5_DIR
                        Path to a local ProtT5 model directory for offline embedding generation. Use this option if the system cannot download the model from the
                        internet. (default: None)
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

## 4 Training

### 4.1 Data Preparation

**Dataset from the paper**

Due to GitHub limitations, the datasets are hosted on [Google Drive link](https://drive.google.com/drive/folders/1-yypLh8dzLW_AJ0MJJCzmDDEE2XBSS6F?usp=drive_link).

**Preparing your own dataset**

If you want to retrain our model using a new dataset, you need to prepare protein sequence FASTA files in the following format:

- Add host information in the format of ` #<host>` at the end of each FASTA definition line for the program to extract labels. For example:

```
>GCF_944325205_gene_1582 #Desulfovibrionaceae
MADFDLAYAPVSKWEGGWTHDSGDKGGETFRGCARNFFPNEPIWPVIDREKSHPSYKQGK
AAFSAHLMGIPSLTGCVKGWYRKEWWDKLGLERFDQIVADELFEQAVNLGKAGMGRYLQR
LCNAFNWRKDGSADGARLFDDLQTDGVVGPKTLSALSIVLSRNDARRIVHLMNCMQGAHY
```

- The host information that can be recognized can be found in the files gut_prophages_info.py and multi_taxonomic_levels_info.py

### 4.2 Training

You can use the following programs to retrain the model or reproduce results using either the data from our paper or your own data.

If you want to retrain the model on the multi-taxonomic levels dataset, you can use the `vhh-train-multi` command.

For example:

```
vhh-train-multi \
--train_protein ./data/multi_taxonomic_levels/tail/family/train_protein.fasta \
--train_dna ./data/multi_taxonomic_levels/tail/family/train_dna.fasta \
--val_protein ./data/multi_taxonomic_levels/tail/family/val_protein.fasta \
--val_dna ./data/multi_taxonomic_levels/tail/family/val_dna.fasta \
--type tail \
--level family
```


Run `vhh-train-multi -h` to view the command-line help message.

```
$ vhh-train-multi -h
usage: train_multi_taxonomic_levels.py [-h] --train_protein TRAIN_PROTEIN --train_dna TRAIN_DNA --val_protein VAL_PROTEIN --val_dna VAL_DNA
                                       --level {family,genus,species} --type {tail,lysin} [--output_dir OUTPUT_DIR]
                                       [--embedding_dir EMBEDDING_DIR] [--prott5_dir PROTT5_DIR]

Train the Viral-Host-Hunter model across multiple taxonomic levels. This script trains a multi-modal classifier using tail or lysin proteins
and their corresponding DNA sequences to predict bacterial hosts at the family, genus, or species level.

optional arguments:
  -h, --help            show this help message and exit
  --train_protein TRAIN_PROTEIN
                        Path to the training protein FASTA file.
  --train_dna TRAIN_DNA
                        Path to the corresponding training DNA FASTA file.
  --val_protein VAL_PROTEIN
                        Path to the validation protein FASTA file.
  --val_dna VAL_DNA     Path to the corresponding validation DNA FASTA file.
  --level {family,genus,species}
                        Taxonomic classification level of the prediction model: "family", "genus", or "species".
  --type {tail,lysin}   Protein type used for training: "tail" or "lysin".
  --output_dir OUTPUT_DIR
                        Directory to save trained model. (default: ./model/multi_taxonomic_levels/{type}/{level})
  --embedding_dir EMBEDDING_DIR
                        Directory to save/load precomputed embeddings (refer to prot_embedding.csv and dna_embedding.csv). If embeddings are
                        already available, the model will reuse them to reduce computation time. (default:
                        ./embeddings/multi_taxonomic_levels/{type}/{level})
  --prott5_dir PROTT5_DIR
                        Path to a local ProtT5 model directory for offline embedding generation. Use this option if the system cannot download
                        the model from the internet. (default: None)
```

Similarly, for the gut prophages dataset, you can run the `train_gut_prophages.py` file to train. This script has the same parameters as the one above, which you can refer to.

### 4.3 Prediction

**For evaluation with labels**

The following code requires you to prepare your data according to the requirements in section 4.1, as labels are needed to calculate evaluation metrics.

**For prediction without labels**

If your data has no labels or you just want to get prediction results, you can directly use the program in section 3.1 to obtain prediction results.

For the multi-taxonomic levels dataset, you can use the `vhh-predict-multi` command to implement VHH evaluation on this dataset.

For example:

```
vhh-predict-multi \
--protein_file ./data/multi_taxonomic_levels/tail/family/test_protein.fasta \
--dna_file ./data/multi_taxonomic_levels/tail/family/test_dna.fasta \
--type tail \
--level family \
--precision -1 \
--embedding_name predict \
--result_file results.csv
```


Run `vhh-predict-multi -h` to view the command-line help message.

```
$ vhh-predict-multi -h
usage: predict_multi_taxonomic_levels.py [-h] --protein_file PROTEIN_FILE --dna_file DNA_FILE --result_file RESULT_FILE --level
                                         {family,genus,species} --type {tail,lysin} --precision {95,84,69,-1} --embedding_dir EMBEDDING_DIR
                                         [--model_dir MODEL_DIR] [--prott5_dir PROTT5_DIR]

Run the Viral-Host-Hunter prediction pipeline for multi-taxonomic-level phage host prediction using protein and DNA embeddings.

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
                        Prediction confidence threshold. "95", "84", and "69" correspond to models with different confidence cutoffs; "-1" uses
                        predictions without threshold filtering.
  --embedding_dir EMBEDDING_DIR
                        Directory containing precomputed embedding files.
  --model_dir MODEL_DIR
                        Directory containing trained multi-taxonomic-level models. (default: ./models/multi_taxonomic_levels/{type}/{level})
  --prott5_dir PROTT5_DIR
                        Path to a local ProtT5 model directory for offline embedding generation. Use this option if the system cannot download
                        the model from the internet. (default: None)
```

For the gut prophages dataset, you can use the `vhh-predict-gut` command to implement VHH evaluation on this dataset. This script has the same parameters as the one above, which you can refer to.

```
vhh-predict-gut --protein_file protein.fasta \
 --dna_file cds.fasta \
 --type tail \
 --level family \
 --precision -1 \
 --embedding_name test \
 --result_file results.csv
```

You can also set the confidence threshold of VHH through the `--precision` parameter (optional thresholds are 95%, 84%, 69%) to obtain higher confidence results.

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


## Contact Information

1. Yuehua Ou, ouyuehua2022@email.szu.edu.cn
2. Zihao Lin, 2410103047@mails.szu.edu.cn

## Copyright Information / License

Please see the "LICENSE.txt" file for the copyright information.
