# VirHostHunter: Decrypting viral dark matter through key proteins using large language models

# Introduction

Understanding virus–host interactions is central to microbiome research, viral ecology, and phage therapy development. Yet, the majority of viral sequences in metagenomic datasets remain fragmental and host-unknown, collectively referred to as viral dark matter.

VirHostHunter (VHH) addresses this challenge through a protein-centered, alignment-free framework that predicts bacterial hosts of phages using key proteins such as tails and lysins, without requiring full genomes. By integrating Protein Language Models (PLMs) and Vision Transformers (ViTs), VHH captures functional homology beyond sequence similarity, enabling high-resolution and scalable host prediction.

This repository provides the datasets, model code, and usage accompanying the paper “Decrypting viral dark matter through key proteins using large language models”, supporting analyses and downstream applications in phage discovery and microbiome therapeutics.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [1. Installation](#1-installation)
  - [1.1 Clone the Repository](#11-clone-the-repository)
  - [1.2 Setup Environment](#12-setup-environment)
  - [1.3 Download Pretrained Models](#13-download-pretrained-models)
  - [1.4 Quick Test](#14-quick-test)
- [2 Usage](#2-usage)
  - [2.1 Basic Usage](#21-basic-usage)
  - [2.2 Output Description](#22-output-description)
- [3 Training (Optional)](#3-training-optional)
  - [3.1 Reproducing VirHostHunter Training](#31-reproducing-virhosthunter-training)
    - [3.1.1 Data Preparation](#311-data-preparation)
    - [3.1.2 Model Training](#312-model-training)
    - [3.1.3 Evaluation and Prediction](#313-evaluation-and-prediction)
  - [3.2  Training with Custom Datasets](#32--training-with-custom-datasets)
    - [3.2.1 Prepare Custom Dataset](#321-prepare-custom-dataset)
    - [3.2.2 Update Label Information](#322-update-label-information)
    - [3.2.3 Modify Training Script](#323-modify-training-script)
    - [3.2.4 Modify Predition Script](#324-modify-predition-script)
    - [3.2.5 Model Training, Evaluation and Prediction](#325-model-training-evaluation-and-prediction)
- [Contact Information](#contact-information)
- [License](#license)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 1. Installation

**GPU Recommendation:**
We strongly recommend using a GPU for all steps (embedding generation, training, and prediction) to ensure reasonable performance and accuracy. While `vhh-predict` can run on CPU for the example data, ProtT5 execution is extremely slow on CPU and we cannot guarantee numerical precision or stability in this mode.
In our case, we used an **NVIDIA GeForce RTX 3090 (24 GiB VRAM)** to generate 1024-dimensional embeddings and perform model training/prediction.

## 1.1 Clone the Repository

```bash
git clone https://github.com/YuehuaOu/Viral-Host-Hunter
cd Viral-Host-Hunter
```

## 1.2 Setup Environment

VirHostHunter was developed and tested with **Python 3.8, PyTorch 2.4.0, and CUDA 11.8.**
To ensure a smooth installation and proper functionality, we recommend creating a dedicated virtual environment and installing the required dependencies:

```bash
# 1. Create a Python 3.8 environment
conda create -n VHH python=3.8

# 2. Activate the environment
conda activate VHH

# 3. Install PyTorch compatible with CUDA 11.8
## Option 1: Conda
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
## Option 2: Pip
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Install 
## Option 1: 
conda install -c wlhuang viral-host-hunter
## Option 2: 
conda install -c https://conda.anaconda.org/wlhuang viral-host-hunter
```

> **Notes**:
>
> - If your system has a different CUDA version, refer to [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) to find the compatible installation command.
> - A Common error: This error indicates a mismatch between your installed PyTorch and CUDA versions. Reinstall PyTorch with the appropriate CUDA toolkit for your GPU.
>
>   ```
>   RuntimeError: CUDA error: no kernel image is available for execution on the device
>   ```

## 1.3 Download Pretrained Models

Pretrained models can be downloaded from our [model repository](https://zenodo.org/records/17340381):

```bash
wget https://zenodo.org/records/17340381/files/models.zip
unzip models.zip
```

When running the `vhh-predict` command, specify the path to the downloaded model directory using the `--model_dir` parameter.

VirHostHunter also requires the pretrained **ProtT5-XL-UniRef50** model for generating protein embeddings:：

- If your machine has internet access, the model will be downloaded automatically at runtime.
- For offline use, manually download the files from [Rostlab/prot_t5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main) to a local directory.
  **Note:** You only need `pytorch_model.bin` (not the other `.bin` files) along with the remaining files. Then, specify your local directory path using the `--prott5_dir` parameter.

## 1.4 Quick Test

Example data and scripts are provided in the `examples/` directory for quick verification of a successful installation. Please include the `--model_dir` parameter in the command to specify the location of the pretrained models:

```bash
vhh-predict \
--protein ./examples/gut/tail/protein.fasta \
--dna ./examples/gut/tail/dna.fasta \
--phage_type gut --seq_type tail \
--embedding_dir ./examples/embedding \
--output_dir ./examples/output \
--model_dir <path_to_models>
```

For detailed descriptions of all command-line parameters, see the Uasage section below.

# 2 Usage

## 2.1 Basic Usage

Use the `vhh-predict` command to perform viral host prediction with the pretrained model.
A runnable example is provided in `example/run_example.sh`.

You can check all available command-line options by running:

```
$ vhh-predict -h
usage: vhh-predict [-h] --protein PROTEIN --dna DNA --seq_type {tail,lysin} [--phage_type {gut,environment}]
                   [--level {all,family,genus,species}] [--model_dir MODEL_DIR] [--embedding_dir EMBEDDING_DIR]
                   [--output_dir OUTPUT_DIR] [--prott5_dir PROTT5_DIR] [--lineage]

Run the Viral-Host-Hunter prediction pipeline for viral host identification based on protein and DNA sequences.

optional arguments:
  -h, --help            Show this help message and exit
  --protein PROTEIN     Path to the input protein FASTA file.
  --dna DNA             Path to the corresponding DNA FASTA file.
  --seq_type {tail,lysin}
                        Protein type used for prediction: "tail" or "lysin".
  --phage_type {gut,environment}
                        Phage source type: "gut" for intestinal phages, or "environment" for environmental phages. (default: gut)
  --level {all,family,genus,species}
                        Taxonomic level: "all", "family", "genus", or "species". (default: all)
  --model_dir MODEL_DIR
                        Directory containing the pretrained Viral-Host-Hunter models.
  --embedding_dir EMBEDDING_DIR
                        Directory to save or load precomputed embeddings (prot_embedding.csv, dna_embedding.csv).
                        Existing embeddings will be reused to speed up prediction. (default: ./embeddings)
  --output_dir OUTPUT_DIR
                        Directory to save prediction results. (default: ./output)
  --prott5_dir PROTT5_DIR
                        Path to a local ProtT5 model directory for offline use. Required if the system has no internet access.
  --lineage             Append host lineage information to the output.
```

## 2.2 Output Description

Prediction results are saved to `$OUTPUT_DIR/predict_result.xlsx` which includes three sheets for family, genus, and species-level predictions.

Each sheet follows the structure below:

- Columns 1–2: Input protein and DNA sequence ID
- Columns 4–7: Predicted hosts at different confidence thresholds (no threshold, 69%, 84%, 95%)

| Protein_Desc    | DNA_Desc    | No_Threshold     | Confidence_69%   | Confidence_84%   | Confidence_95%   |
| --------------- | ----------- | ---------------- | ---------------- | ---------------- | ---------------- |
| tail_1 #protein | tail_1 #dna | Eubacteriaceae   | Eubacteriaceae   | Unknown          | Unknown          |
| tail_2 #protein | tail_2 #dna | Eubacteriaceae   | Eubacteriaceae   | Eubacteriaceae   | Eubacteriaceae   |
| tail_3 #protein | tail_3 #dna | Xanthomonadaceae | Xanthomonadaceae | Xanthomonadaceae | Xanthomonadaceae |
| tail_4 #protein | tail_4 #dna | Xanthomonadaceae | Xanthomonadaceae | Xanthomonadaceae | Xanthomonadaceae |
| tail_5 #protein | tail_5 #dna | Bacteroidaceae   | Bacteroidaceae   | Unknown          | Unknown          |

If the `--lineage` option is applied, an additional set of columns containing the full host lineage will be included in the output.

# 3 Training (Optional)

## 3.1 Reproducing VirHostHunter Training

### 3.1.1 Data Preparation

To retrain VirHostHunter using the same datasets as in our paper, download and extract the training data using the following commands.
The datasets are pre-split into training, validation, and test sets according to the procedures described in the publication.

```bash
wget https://zenodo.org/records/17340915/files/data.zip
unzip data.zip
```

### 3.1.2 Model Training

Models can be trained for different datasets using the provided scripts:

- `vhh-train-gut`: training models on the gut prophage dataset
- `vhh-train-multi`: training models on the environmental phage dataset

Example command for training on the gut prophage dataset:

```bash
vhh-train-gut \
--train_protein <path_to_data>/gut_prophages/lysin/species/train_protein.fasta \
--train_dna <path_to_data>/gut_prophages/lysin/species/train_dna.fasta \
--val_protein <path_to_data>/gut_prophages/lysin/species/val_protein.fasta \
--val_dna <path_to_data>/gut_prophages/lysin/species/val_dna.fasta \
--type lysin \
--level species
```

Use `--type` and `--level` to train models for different phage types and taxonomic levels. Run `vhh-train-gut -h` to view the command-line help message:

```
$ vhh-train-gut -h
usage: vhh-train-gut [-h] --train_protein TRAIN_PROTEIN --train_dna TRAIN_DNA --val_protein VAL_PROTEIN --val_dna VAL_DNA --level {family,genus,species} --type
                     {tail,lysin} [--output_dir OUTPUT_DIR] [--embedding_dir EMBEDDING_DIR] [--prott5_dir PROTT5_DIR]

Train the Viral-Host-Hunter model for gut-associated phage host prediction. This script trains a multi-modal classifier using tail or lysin proteins and their
corresponding DNA sequences to predict bacterial hosts at the family, genus, or species level.

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
  --type {tail,lysin}   Protein type used for model training: "tail"or "lysin".
  --output_dir OUTPUT_DIR
                        Directory to save trained model. (default: ./model/gut_prophages/{type}/{level})
  --embedding_dir EMBEDDING_DIR
                        Directory to save/load precomputed embeddings (refer to prot_embedding.csv and dna_embedding.csv). If embeddings are already available,
                        the model will reuse them to reduce computation time. (default: ./embeddings/gut_prophages/{type}/{level})
  --prott5_dir PROTT5_DIR
                        Path to a local ProtT5 model directory for offline embedding generation. Use this option if the system cannot download the model from
                        the internet. (default: None)
```

Similarly, for the environmental phage dataset, you can run the `vhh-train-multi` command to train. For example:

```bash
vhh-train-multi \
--train_protein <path_to_data>/multi_taxonomic_levels/tail/family/train_protein.fasta \
--train_dna <path_to_data>/multi_taxonomic_levels/tail/family/train_dna.fasta \
--val_protein <path_to_data>/multi_taxonomic_levels/tail/family/val_protein.fasta \
--val_dna <path_to_data>/multi_taxonomic_levels/tail/family/val_dna.fasta \
--type tail \
--level family
```

### 3.1.3 Evaluation and Prediction

After training, models are evaluated by using the test datasets to calculate the metrics .

**For the gut prophages dataset**, evaluation is performed using the `vhh-predict-gut` command:

```bash
vhh-predict-gut --protein_file protein.fasta \
 --dna_file cds.fasta \
 --type tail \
 --level family \
 --precision -1 \
 --embedding_name test \
 --result_file results.csv
```

Tips:

- `--precision` sets the confidence threshold (95%, 84%, 69%, or -1 for no filtering).
- If a custom model path was specified during training, the same path should be provided with the `--model_dir` option during prediction.

Run `vhh-predict-gut -h` to view all parameters and help message.

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

**For the environmental phage dataset**, evaluation is conducted using the `vhh-predict-multi` command:

```bash
vhh-predict-multi \
--protein_file <path_to_data>/multi_taxonomic_levels/tail/family/test_protein.fasta \
--dna_file <path_to_data>/multi_taxonomic_levels/tail/family/test_dna.fasta \
--type tail \
--level family \
--precision -1 \
--embedding_name predict \
--result_file results.csv
```

Similarily, run `vhh-predict-multi -h` to view all parameters and help message.

## 3.2  Training with Custom Datasets

To retrain VirHostHunter using a custom dataset:

### 3.2.1 Prepare Custom Dataset

- Provide protein and DNA FASTA files.
- Add host labels in the format #`<host>` at the end of each FASTA header. For example:

```fasta
>GCF_944325205_gene_1582 #Desulfovibrionaceae
MADFDLAYAPVSKWEGGWTHDSGDKGGETFRGCARNFFPNEPIWPVIDREKSHPSYKQGK
AAFSAHLMGIPSLTGCVKGWYRKEWWDKLGLERFDQIVADELFEQAVNLGKAGMGRYLQR
LCNAFNWRKDGSADGARLFDDLQTDGVVGPKTLSALSIVLSRNDARRIVHLMNCMQGAHY
```

- Each protein sequence must has the corresponding DNA sequence, using the same sequence identifier and host tag. For example:

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

### 3.2.2 Update Label Information

Define labels for the new dataset in a separate Python file (e.g., new_data_info.py) following the structure of `multi_taxonomic_levels_info.py` for providing labels corresponding to the customized dataset.

### 3.2.3 Modify Training Script

Adapt the training script for the new dataset, using `train_multi_taxonomic_levels.py` as a template. In particular, rReplace the label import statement with the new label file.

```
from .multi_taxonomic_levels_info import info
```

with:

```
from .new_data_info import info
```

### 3.2.4 Modify Predition Script

Apply analogous modifications to the prediction script as you did for training. Use `predict_multi_taxonomic_levels.py` as a reference, and replace the label import with the customized label file.

### 3.2.5 Model Training, Evaluation and Prediction

Follow the same procedures as described in Sections **4.1.2 Model Training** and **4.1.3 Evaluation and Prediction**.

# Contact Information

1. Yuehua Ou, ouyuehua2022@email.szu.edu.cn
2. Zihao Lin, 2410103047@mails.szu.edu.cn
3. Min Li, limin19@mails.ucas.edu.cn
4. Bo Xing, xingbo@genomics.cn

# License

Viral-Host-Hunter is licensed under the **GNU General Public License v3.0** - see the LICENSE.txt file for full details.
