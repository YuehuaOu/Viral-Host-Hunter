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
  - [2.1 Parameters](#21-parameters)
  - [2.2 Output Description](#22-output-description)
- [3 Training (Optional)](#3-training-optional)
  - [3.1 Reproducing VirHostHunter Training](#31-reproducing-virhosthunter-training)
    - [3.1.1 Data Preparation](#311-data-preparation)
    - [3.1.2 Model Training](#312-model-training)
    - [3.1.3 Prediction and Evaluation](#313-prediction-and-evaluation)
  - [3.2  Training with Custom Datasets](#32--training-with-custom-datasets)
    - [3.2.1 Prepare Custom Dataset](#321-prepare-custom-dataset)
    - [3.2.2 Update Label Information](#322-update-label-information)
    - [3.2.3 Modify Training Script](#323-modify-training-script)
    - [3.2.4 Modify Predition Script](#324-modify-predition-script)
    - [3.2.5 Model Training, Prediction and Evaluation](#325-model-training-prediction-and-evaluation)
- [4 Troubleshooting](#4-troubleshooting)
- [Contact Information](#contact-information)
- [License](#license)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 1. Installation

**GPU Recommendation:**

We strongly recommend using a GPU for all steps (embedding generation, training, and prediction) to ensure reasonable performance and accuracy. While `vhh-predict` can run on CPU for the example data, ProtT5 execution is extremely slow on CPU and we cannot guarantee numerical precision or stability in this mode.

In our case, we used an **NVIDIA GeForce RTX 3090 (24 GiB VRAM)** to generate 1024-dimensional embeddings and perform model training/prediction.

Follow the steps below to complete the installation. We also provide demonstration videos showing successful installation and usage on multiple platforms: https://www.youtube.com/watch?v=qu0Hw80xRpY

🛠️ For any installation issues, feel free to contact us via GitHub issues or email.

## 1.1 Clone the Repository

```bash
git clone https://github.com/YuehuaOu/Viral-Host-Hunter
cd Viral-Host-Hunter
```

## 1.2 Setup Environment

VirHostHunter was developed and tested with **Python 3.9, PyTorch 2.4.0, and CUDA 11.8.**

To ensure a smooth installation and proper functionality, we recommend creating a dedicated virtual environment and installing the required dependencies:

```bash
# Environment setup: Create Python 3.9 venv and activate it
conda create -n VHH python=3.9
conda activate VHH

# Install vhh

## Methods 1: via pip
pip install git+https://github.com/YuehuaOu/Viral-Host-Hunter.git

## Methods 2: via conda
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c bioconda viral-host-hunter
conda install -c conda-forge transformers=4.51 # 5. Install transformers 4.51 (To be included in bioconda viral-host-hunter v0.2.0)

```

> **Notes**:
>
> - If your system has a different CUDA version, refer to [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) to find the compatible installation command.
> - A Common error: This error indicates a mismatch between your installed PyTorch and CUDA versions. Reinstall PyTorch with the appropriate CUDA toolkit for your GPU.
>
>   ```
>   RuntimeError: CUDA error: no kernel image is available for execution on the device
>   ```
> - ⚠️ Please check whether your `transformers` package version is **<= 4.51**. If not, please manually downgrade your `transformers` package, otherwise it may cause errors during use. See Section [4 Troubleshooting](#4-troubleshooting) for more details.

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

Run `./examples/run_example.sh <model_dir> [prott5_dir]` **in the Viral-Host-Hunter directory** to quickly test the installation with the example data. 


```bash
bash ./examples/run_example.sh /path/to/models_dir               # online ProtT5
# or
bash ./examples/run_example.sh /path/to/models_dir /path/to/prott5_dir   # offline ProtT5
```

If the command runs successfully, you should see a result similar to the following:
<p align="center">
  <img src="figures/Run_Example_ScreenShot.png" 
       alt="Example Output" 
       width="70%" 
       style="border-radius: 10px;">
</p>

# 2 Usage

Use the `vhh-predict` command to perform viral host prediction with the pretrained model.

Example:

```bash
vhh-predict \
--protein /path/to/your/protein.fasta \
--dna /path/to/your/dna.fasta \
--seq_type tail \
--model_dir /path/to/models_dir
--phage_type gut
```

## 2.1 Parameters


| Category | Argument | Description | Default / Options |
| :--- | :--- | :--- | :--- |
| **Input** | `--protein` | **(Required)** Path to the protein FASTA file. | - |
| | `--dna` | **(Required)** Path to the corresponding DNA FASTA file. | - |
| | `--seq_type` | **(Required)** Protein type for prediction. | `tail`, `lysin` |
| **Model** | `--model_dir` | **(Required)** Directory containing trained models. | - |
| | `--phage_type` | Phage source environment (see details below). | `gut` (default), `environment` |
| | `--level` | Taxonomic prediction depth. | `all` (default), `family`, `genus`, `species` |
| Output | `--output_dir` | Directory to save prediction results. | `./output` |
| | `--output_format`| File format for results. | `csv`, `tsv`, `xlsx`, `both` |
| | `--lineage` | Flag: Append full lineage columns to output. | *Disabled* |
| Other | `--embedding_dir` | Directory to save/load precomputed embeddings. | `./embeddings` |
| | `--prott5_dir` | Local ProtT5 path for **offline** mode. | - |

> **Note on `--phage_type`:**
> - `gut`: Uses the model trained on the **gut_prophages** dataset (disease-associated datasets in the paper).
> - `environment`: Uses the model trained on the **multi_taxonomic_levels** dataset (multi-taxonomic datasets in the paper).






## 2.2 Output Description

Prediction results land in $OUTPUT_DIR. All outputs share the same column layout:

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

*For more detailed parameter information, you can always run:*
```bash
vhh-train-gut -h
vhh-train-multi -h
```

Command Examples：

```bash
# Train models on the gut prophage dataset (lysin phage, species level)
vhh-train-gut \
--train_protein <path_to_data>/gut_prophages/lysin/species/train_protein.fasta \
--train_dna <path_to_data>/gut_prophages/lysin/species/train_dna.fasta \
--val_protein <path_to_data>/gut_prophages/lysin/species/val_protein.fasta \
--val_dna <path_to_data>/gut_prophages/lysin/species/val_dna.fasta \
--type lysin \
--level species

# Train models on the environmental phage dataset (tail phage, family level)
vhh-train-multi \
--train_protein <path_to_data>/multi_taxonomic_levels/tail/family/train_protein.fasta \
--train_dna <path_to_data>/multi_taxonomic_levels/tail/family/train_dna.fasta \
--val_protein <path_to_data>/multi_taxonomic_levels/tail/family/val_protein.fasta \
--val_dna <path_to_data>/multi_taxonomic_levels/tail/family/val_dna.fasta \
--type tail \
--level family
```
Tips:
- Use `--type` and `--level` to train models for different phage types and taxonomic levels. 


### 3.1.3 Prediction and Evaluation

After training, models are evaluated by using the test datasets to calculate the metrics:

- `vhh-predict-gut`: predicts and evaluates for the gut prophage dataset
- `vhh-predict-multi`: predicts and evaluates hosts for the environmental phage dataset

*For more detailed parameter information, you can always run:*
```bash
vhh-predict-gut -h
vhh-predict-multi -h
```

Command Examples：

```bash
# Predict and evaluate for the gut prophage dataset
vhh-predict-gut \
--protein_file <path_to_data>/gut_prophages/lysin/species/test_protein.fasta \
--dna_file <path_to_data>/gut_prophages/lysin/species/test_dna.fasta \
--type lysin \
--level species \
--precision -1 \
--result_dir /path/to/output_dir

# Predict and evaluate for the environmental phage dataset
vhh-predict-multi \
--protein_file <path_to_data>/multi_taxonomic_levels/tail/family/test_protein.fasta \
--dna_file <path_to_data>/multi_taxonomic_levels/tail/family/test_dna.fasta \
--type tail \
--level family \
--precision -1 \
--result_dir /path/to/output_dir
```

Tips:

- `--precision` sets the confidence threshold (95%, 84%, 69%, or -1 for no filtering).
- If a custom model path was specified during training, the same path should be provided with the `--model_dir` option during prediction.
- The directory for `--result_file` needs to be created in advance. We will fix this in the future to create it automatically.


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

### 3.2.5 Model Training, Prediction and Evaluation

Follow the same procedures as described in Sections **3.1.2 Model Training** and **3.1.3 Prediction and Evaluation**.

# 4 Troubleshooting

This chapter summarizes several issues reported by users during actual usage, along with explanations and suggested solutions.



**torch.load Safety Check Error**

Example error message:

```
in check_torch_load_is_safe
raise ValueError(
ValueError: Due to a serious vulnerability issue in torch.load, even with weights_only=True, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.
See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
```
This error occurs because **`transformers >= 4.52`** introduces a mandatory safety check when calling `torch.load`. To avoid this issue, `torch >= 2.6` is required.  

However, this project is currently based on **`torch = 2.4`**, so the recommended solution is to **downgrade `transformers` to version `4.51` or lower**:

```
conda install -c conda-forge transformers=4.51
# or
mamba install transformers=4.51 -c conda-forge
```

An `environment.yml` file is provided to help users verify and align their dependency versions. And We will support `torch==2.6` in the next release.


# Contact Information

1. Zihao Lin, 2410103047@mails.szu.edu.cn
2. Min Li, limin19@mails.ucas.edu.cn
3. Yuehua Ou, ouyuehua2022@email.szu.edu.cn
4. Bo Xing, xingbo@genomics.cn

# License

Viral-Host-Hunter is licensed under the **GPL-3.0** - see the LICENSE.txt file for full details.
