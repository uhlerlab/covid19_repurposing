#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 18:02:18 2020

@author: louis.cammarata
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 17:33:01 2020

@author: louis.cammarata
"""

# Imports

import numpy as np
import networkx as nx
import pickle
import logging
logger = logging.getLogger(__name__)
import neighborhoods as nbh


# Test add_edge_confidence
network_selected = pickle.load(open("../Save/network_selected_with_drug_info.pickle", "rb"))
nbh.add_edge_confidence(network_selected)

# Test plot_neighborhood_subnetwork
min_prize = np.floor(min(nx.get_node_attributes(network_selected,'log2FC_blanco').values()))
max_prize = np.floor(max(nx.get_node_attributes(network_selected,'log2FC_blanco').values()))+1
vmin_nodes = -max(abs(min_prize),abs(max_prize))
vmax_nodes = max(abs(min_prize),abs(max_prize))
protein_center = 'HDAC1'

nbh.plot_neighborhood_subnetwork(protein_center,
                             network_selected,
                             vmin_nodes,
                             vmax_nodes,
                             vmin_edges = 0,
                             vmax_edges = 1,
                             removeUBC = True, 
                             cthreshold = 0.53, 
                             nodesize = 1000,
                             save=False)

