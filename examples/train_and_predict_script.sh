# Example 1.1 run train_gut_prophages.py (lysin, species)
vhh-train-gut \
--train_protein ./data/gut_prophages/lysin/species/train_protein.fasta \
--train_dna ./data/gut_prophages/lysin/species/train_dna.fasta \
--val_protein ./data/gut_prophages/lysin/species/val_protein.fasta \
--val_dna ./data/gut_prophages/lysin/species/val_dna.fasta \
--type lysin \
--level species

# Example 1.2 run predict_gut_prophages.py (lysin, species, 95)
vhh-predict-gut \
--protein_file ./data/gut_prophages/lysin/species/test_protein.fasta \
--dna_file ./data/gut_prophages/lysin/species/test_dna.fasta \
--type lysin \
--level species \
--result_file predict_result_gut_lysin_species.csv \
--precision 95

# Example 2.1 run train_multi_taxonomic_levels.py (tail, family)
vhh-train-multi \
--train_protein ./data/multi_taxonomic_levels/tail/family/train_protein.fasta \
--train_dna ./data/multi_taxonomic_levels/tail/family/train_dna.fasta \
--val_protein ./data/multi_taxonomic_levels/tail/family/val_protein.fasta \
--val_dna ./data/multi_taxonomic_levels/tail/family/val_dna.fasta \
--type tail \
--level family

# # Example 2.2 run predict_multi_taxonomic_levels.py (tail, family, 84)
vhh-predict-multi \
--protein_file ./data/multi_taxonomic_levels/tail/family/test_protein.fasta \
--dna_file ./data/multi_taxonomic_levels/tail/family/test_dna.fasta \
--type tail \
--level family \
--result_file predict_result_multi_tail_family.csv \
--precision 84

