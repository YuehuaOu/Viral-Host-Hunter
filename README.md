# Decrypting Viral Dark Matter with Predictive Framework and Therapeutic Implications

## 1. Introduction
This repository contains source data and code for paper "Decrypting Viral Dark Matter with Predictive Framework and Therapeutic Implications".

## 2. Installation
```
python=3.8
pytorch=1.10.1
biopython=1.81
transformers=4.38.2
SentencePiece=0.2.0
scikit-learn=1.3.2
```
Notice:
1. You need install pretrained language modoel **ProtT5-XL-UniRef50**, the link is provided on [ProtT5-XL-U50](https://github.com/agemagician/ProtTrans#models).
2. You need to modify the model file path in the ``embedding.py`` of the project to the address where the ProtT5 model you downloaded is located.

## 3. Requirments
In order to run successfully, the embedding of ProtT5-XL-UniRef50 requires GPU. We utilized an NVIDIA GeForce RTX 3090 with 24GiB to embed peptide or protein sequences to 1024-dimensional vector.

## 4. Usage
### 4.1 Train
If you want to retrain the model on the multi taxonomic levels dataset, you could run ``train_multi_taxonomic_levels.py`` file to achieve.
```
python train_multi_taxonomic_levels.py --train_protein train_protein.fasta --train_dna train_dna.fasta --val_protein val_protein.fasta --val_dna val_dna.fasta --type tail --level family
```
Similarly, for the gut prophages dataset, you could run ``train_gut_prophages.py`` file to achieve.
```
python train_gut_prophages.py --train_protein train_protein.fasta --train_dna train_dna.fasta --val_protein val_protein.fasta --val_dna val_dna.fasta --type tail --level family
```

### 4.2 Predict
For multi taxonomic levels dataset, you could run ``predict_multi_taxonomic_levels.py`` file to implement VHH evaluation on this dataset.
```
python predict_multi_taxonomic_levels.py --protein_file protein.fasta --dna_file cds.fasta --type tail --level family --precision -1 --embedding_name test --result_file results.csv
```

For gut prophages dataset, you could run ``predict_gut_prophages.py`` file to implement VHH evaluation on this dataset.
```
python predict_gut_prophages.py --protein_file protein.fasta --dna_file cds.fasta --type tail --level family --precision -1 --embedding_name test --result_file results.csv
```

If you want to use the VHH model to predict some samples, you can set the accuracy of VHH (optional accuracy is 95%, 84%, 69%)

```
python predict_multi_taxonomic_levels.py --protein_file protein.fasta --dna_file cds.fasta --type tail --level family --precision 95 --embedding_name predict --result_file results.csv
```
or
```
python predict_gut_prophages.py --protein_file protein.fasta --dna_file cds.fasta --type tail --level family --precision 95 --embedding_name predict --result_file results.csv
```

