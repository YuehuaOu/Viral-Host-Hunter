#!/bin/bash
vhh-predict \
--protein ./examples/gut/tail/protein.fasta \
--dna ./examples/gut/tail/dna.fasta \
--phage_type gut --seq_type tail \
--lineage \
--embedding_dir ./examples/embedding \
--output_dir ./examples/output \
--model_dir <path_to_models>
