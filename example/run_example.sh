#!/bin/bash
vhh-predict \
--protein ./example/gut/tail/protein.fasta \
--dna ./example/gut/tail/dna.fasta \
--phage_type gut --seq_type tail --model_dir ./models \
--embedding_dir ./example/embedding \
--output_dir ./example/output \
--lineage \