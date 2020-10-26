#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 15:31:59 2020

@author: louis.cammarata
"""

# Imports
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)
import sensitivity_viruspartners as svp

# Load datasets
SARS_targets_file = "../Data/SARSCov_targets_df.tsv"
SARS_targets_df = pd.read_csv(SARS_targets_file, sep='\t')
interacting_genes = set(SARS_targets_df['protein2'])
prize_file = "../Save/terminals_ppi_analysis.tsv"
prizes_data = pd.read_csv(prize_file, sep='\t')
prized_names = prizes_data['name']


# Test run_prize_sensitivity_analysis
interactome_file_name = "iRefIndex_v14_MIScore_interactome_C9.costs.allcaps.txt"
prize_file_name = "terminals_ppi_analysis.tsv"
graph_params =  {
                "noise": 0.0, 
                "dummy_mode": "terminals", 
                "exclude_terminals": False, 
                "seed": 1,
                "pruning": 'strong',
                "verbosity_level": 0,
                "w": 1.4,
                "b": 40,
                "g": 0
                }
virus_interacting_genes = interacting_genes
terminals = list(prizes_data['name'])
terminal_partners = set(terminals).intersection(virus_interacting_genes)
P_list = np.arange(0,0.0016,0.0001)
networks_dict = svp.run_prize_sensitivity_analysis(interactome_file_name, prize_file_name, graph_params, virus_interacting_genes, terminal_partners, P_list)

# Test add_metadata    
networks_dict = svp.add_metadata(networks_dict, virus_interacting_genes)

# Test make_summary
n_terminals = 162
networks_summary_df = svp.make_summary(networks_dict, n_terminals)
print(networks_summary_df.head())























