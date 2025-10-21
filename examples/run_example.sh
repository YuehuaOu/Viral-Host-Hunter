#!/bin/bash
vhh-predict \
--protein ./examples/gut/tail/protein.fasta \
--dna ./examples/gut/tail/dna.fasta \
--phage_type gut --seq_type tail --model_dir ./models \
--embedding_dir ./examples/embedding \
--output_dir ./examples/output \
--lineage