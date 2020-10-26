#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 13:21:43 2020

@author: louis.cammarata
"""

# Imports
import numpy as np
import pandas as pd
import networkx as nx
import logging
logger = logging.getLogger(__name__)
import OmicsIntegrator as oi

import sensitivity_analysis as sensitivity


# Test import_virus_partners
virus_partners_file_name = "../Data/SARSCov_targets_df.tsv"
interacting_genes = sensitivity.import_virus_partners(virus_partners_file_name)

# Test run_sensitivity_analysis
interactome_file_name = "../Data/iRefIndex_v14_MIScore_interactome_C9.costs.allcaps.txt"
prize_file_name = "../Save/terminals_ppi_analysis.tsv"
graph_params = {
            "noise": 0.0, 
            "dummy_mode": "terminals", 
            "exclude_terminals": False, 
            "seed": 1,
            "pruning": 'strong',
            "verbosity_level": 0
            }
W_list = np.linspace(start = 0.2, stop = 2, num = 10)
B_list = np.array([5., 10., 15., 20., 25., 30., 35., 40., 45., 50.])
networks_dict = sensitivity.run_sensitivity_analysis(interactome_file_name, prize_file_name, graph_params, W_list, B_list, G=0)

# Test add_metadata
virus_interacting_genes = import_virus_partners(virus_partners_file_name)
networks_dict = sensitivity.add_metadata(networks_dict, virus_interacting_genes)

# Test make_summary
n_terminals = 162
networks_summary_df = sensitivity.make_summary(networks_dict, n_terminals, g=0)
networks_summary_df.head()

# Test create_matrix_gene_overlap_between_networks    
mat_allnodes = sensitivity.create_matrix_gene_overlap_between_networks(networks_summary_df)

# Test create_matrix_terminal_overlap_between_networks
mat_terminals = sensitivity.create_matrix_terminal_overlap_between_networks(networks_summary_df)

# Test create_matrix_sars_overlap_between_networks    
mat_sars = sensitivity.create_matrix_sars_overlap_between_networks(networks_summary_df)
