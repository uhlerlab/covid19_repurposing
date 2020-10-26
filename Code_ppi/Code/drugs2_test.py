#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 16:54:17 2020

@author: louis.cammarata
"""


# Import
import pickle
import logging
logger = logging.getLogger(__name__)
import drugs2

# Test load_drug_target_data    
drug_target_file_name = '../Data/drug.target.interaction.tsv'
drugcentral_df = drugs2.load_drug_target_data(drug_target_file_name, aff_cst_thresh=5)

# Test load_embedded_drugs
embedded_drugs_file_name = '../Data/drug_lists_Adit/final_A549_drug_correlations.csv'
bestdrugs_df = drugs2.load_embedded_drugs(embedded_drugs_file_name, lower_corr_thresh=-2)


# Test add_drug_info_to_selected_network
network_selected = pickle.load(open('../Save/network_selected.pickle', "rb"))
targets_and_drugs_df = drugcentral_df.merge(bestdrugs_df, on = 'drug', how = 'inner')
network_selected = drugs2.add_drug_info_to_selected_network(network_selected, targets_and_drugs_df)

# Test drug_targets_in_selected_network
drug_targets_df = drugs2.drug_targets_in_selected_network(network_selected)

