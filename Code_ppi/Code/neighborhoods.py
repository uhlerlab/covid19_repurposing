#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 17:33:01 2020

@author: louis.cammarata
"""

# Imports

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import logging
logger = logging.getLogger(__name__)
import OmicsIntegrator as oi


def add_edge_confidence(network_selected):
    """ Adds edge confidence to network attribute
    
    Args:
        network_selected: NetworkX network with cost information
    
    Returns:
        void
    """
    cost = nx.get_edge_attributes(network_selected,'cost')
    edge_costs = [cost[edge] for edge in list(network_selected.edges)]
    edge_confidences = {edge: (max(edge_costs)-cost[edge])/(max(edge_costs)-min(edge_costs)) for edge in list(network_selected.edges)}
    nx.set_edge_attributes(network_selected, edge_confidences, 'confidence')


def plot_neighborhood_subnetwork(protein_center,
                                 network_selected,
                                 vmin_nodes,
                                 vmax_nodes,
                                 vmin_edges = 0,
                                 vmax_edges = 1,
                                 removeUBC = True, 
                                 cthreshold = 1, 
                                 nodesize = 1000,
                                 save=False):
    
    # Handle UBC
    network_selected_new = network_selected.copy()
    if removeUBC == True:
        network_selected_new.remove_node('UBC')

    # Apply cost threshold
    cost = nx.get_edge_attributes(network_selected_new,'cost')
    cost_threshold = cthreshold
    expensive_edges = list([edge for edge in list(network_selected_new.edges) if cost[edge]>cost_threshold])
    network_selected_new.remove_edges_from(expensive_edges)

    # Find neighbors 2 hops away of protein_center
    neighborhood_set = set()
    for node in network_selected_new.neighbors(protein_center):
        neighborhood_set = neighborhood_set.union(set({node}))
        for node2 in network_selected_new.neighbors(node):
            neighborhood_set = neighborhood_set.union(set({node2}))
    not_in_neighborhood = set(network_selected_new.nodes()).difference(neighborhood_set)

    # Create subnetwork
    neighborhood_net = network_selected_new.copy()
    neighborhood_net.remove_nodes_from(not_in_neighborhood)
    neighborhood_df = oi.get_networkx_graph_as_dataframe_of_nodes(neighborhood_net)
    
    # Define type of nodes
    node_ego = set({protein_center})
    node_viruspartners = set(neighborhood_df[neighborhood_df['interact_sars_bool']==True].index).difference(node_ego)
    node_druggable = set(neighborhood_df[neighborhood_df['druggable']==True].index)
    node_terminals = set(neighborhood_df[neighborhood_df['prize']>0.001].index)
    colors_terminals = neighborhood_df.loc[neighborhood_df['prize']>0.001]['log2FC_blanco']
    steiner_nodes = set(neighborhood_net.nodes()).difference(node_viruspartners.union(node_terminals).union(node_ego))

    # edge colors
    edge_confidences = nx.get_edge_attributes(neighborhood_net,'confidence')
    # color map
    colors = [(1, 0.5, 1), (0, 0.21, 1)]
    cmap = LinearSegmentedColormap.from_list('test', colors, N=20)

    # Draw
    plt.figure(figsize = (30,20))
    pos = nx.layout.kamada_kawai_layout(neighborhood_net,
                                        weight=None,
                                        scale=10)
    # Draw terminals 
    terminals = nx.draw_networkx_nodes(neighborhood_net, 
                                       pos,
                                       nodelist=sorted(list(node_terminals)),
                                       node_color=list(colors_terminals),
                                       node_size=nodesize,
                                       node_sha6e='o',
                                       alpha=1,
                                       cmap=plt.cm.RdBu_r,
                                       vmin=vmin_nodes, 
                                       vmax=vmax_nodes)
    # Draw all other nodes
    nx.draw_networkx_nodes(neighborhood_net, pos,
                           nodelist=steiner_nodes,
                           node_color='grey',
                           node_size=nodesize,
                           node_shape='o',
                           alpha=0.4)
    
    # Draw druggable nodes 
    nx.draw_networkx_nodes(neighborhood_net, pos,
                           nodelist=node_druggable,
                           node_color='g',
                           node_size=nodesize,
                           node_shape='d',
                           alpha=0.4)
    # Draw partners 
    nx.draw_networkx_nodes(neighborhood_net, pos,
                           nodelist=node_viruspartners,
                           node_color='b',
                           node_size=nodesize,
                           node_shape='s',
                           alpha=0.4)
    # Draw ego 
    nx.draw_networkx_nodes(neighborhood_net, pos,
                           nodelist=[protein_center],
                           node_color='r',
                           node_size=nodesize,
                           node_shape='h',
                           alpha=1)
    # Draw edges
    edges = nx.draw_networkx_edges(neighborhood_net, 
                                   pos, 
                                   width=2.0, 
                                   alpha=0.5,
                                   edge_color=list(edge_confidences.values()),
                                   edge_cmap=cmap,
                                   edge_vmin=vmin_edges, edge_vmax = vmax_edges)
    # Draw labels
    nx.draw_networkx_labels(neighborhood_net,
                            pos,
                            font_size=14,
                            font_weight = 'bold')

    plt.colorbar(edges)
    plt.colorbar(terminals)
    terminals.set_alpha(0.4)
    plt.axis('off')
    
    if save == True:
        plt.savefig('neighborhood'+str(protein_center)+'_P_cthresh'+str(cthreshold)+'.pdf', format='pdf')
    
    plt.show()
