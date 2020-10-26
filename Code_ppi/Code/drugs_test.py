#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:12:45 2020

@author: louis.cammarata
"""

# Imports
import drugs

# Test preprocess_drug_target_data
drug_target_file_name = '../Data/ligand_protein_binding_human.csv'
drugs_with_targets = drugs.preprocess_drug_target_data(drug_target_file_name)
print(len(drugs_with_targets))

# Test drugs_from_embedding
drugs_from_embedding_file_name = '../Data/original_space_series16_correlations_after_clustering.txt'
drugs_from_embedding = drugs.drugs_from_embedding(drugs_from_embedding_file_name)
print(len(drugs_from_embedding))

# Test drugs_from_embedding_with_targets
drugs_from_embedding_with_targets = drugs.drugs_from_embedding_with_targets(drugs_from_embedding, drugs_with_targets)
print(len(drugs_from_embedding_with_targets))
