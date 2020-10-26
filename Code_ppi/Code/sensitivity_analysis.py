#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 12:17:01 2020

@author: louis.cammarata
"""

# Imports
import numpy as np
import pandas as pd
import networkx as nx
import logging
logger = logging.getLogger(__name__)
import OmicsIntegrator as oi


def import_virus_partners(virus_partners_file_name):
    """ Loads data on SARS-Cov-2 human interacting proteins from Gordon et al.
    
    Args:
        virus_partners_file_name: path to virus partners data set
    
    Returns:
        The set of SARS-Cov-2 human interacting proteins
    """
    SARS_targets_df = pd.read_csv(virus_partners_file_name, sep='\t')
    interacting_genes = set(SARS_targets_df['protein2'])
    return(interacting_genes)


def run_sensitivity_analysis(interactome_file_name, prize_file_name, graph_params, W_list, B_list, G=0):
    """ Runs Steiner tree algorithm on the prized interactome with different values of the parameters B, W for a given value of G.
    Stores the results in a dictionary.
    
    Args:
        interactome_file_name: path to the IREF interactome
        prize_file_name: path to the terminal file (prized nodes)
        graph_params: dictionary of graph hyperparameters
        W_list: values of W
        B_list: values of B
        G: value of G
    
    Returns:
        A dictionary of networks corresponding to different (W,B,G) values.
    """
    
    # Build graph
    graph = oi.Graph(interactome_file_name, graph_params)
    
    # Compute range lengths
    lW = len(W_list)
    lB = len(B_list)
    N = lW*lB
    
    # Build Steiner trees for different values of W and B, store results in a dictionary
    networks_dict = {}
    count = 0
    for i in np.arange(lW):
        for j in np.arange(lB):
            # Reset parameters
            hyperparams = {"w": W_list[i], 
                           "b": B_list[j], 
                           "g": G,
                           "pruning": 'strong',
                           "edge_noise": 0, 
                           "dummy_mode": "terminals", 
                           "seed": 0, 
                           "skip_checks": False,
                           "verbosity_level": 0}
            graph._reset_hyperparameters(hyperparams)
            graph.prepare_prizes(prize_file_name)
            # Run PCSF
            vertex_indices, edge_indices = graph.pcsf()
            forest, augmented_forest = graph.output_forest_as_networkx(vertex_indices, edge_indices)
            # Store result in network
            key = "w_"+str(W_list[i])+"_b_"+str(B_list[j])
            networks_dict[key] = augmented_forest
            # Print progress
            count = count+1
            print('Progress= '+str(count/N))
            
    return(networks_dict)


def add_metadata(networks_dict, virus_interacting_genes):
    """ Add metadata on SARS-Cov-2 interacting genes and log2FC between A549-ACE2 w/ virus versus
    A549-ACE2 w/o virus from Blanco et al
    
    Args:
        networks_dict: dictionary of networks, output of run_sensitivity_analysis
        virus_interacting_genes: set of genes whose corresponding proteins interact with SARS-Cov-2
    
    Returns:
        A dictionary of networks corresponding to different (W,B,G) values enriched with metadata.
    """
    
    
    # Add interact_sars attribute for each network
    for network in networks_dict.values():
        interact_sars = {name: str(name in virus_interacting_genes) for name in list(nx.nodes(network))}
        interact_sars_bool = {name: (name in virus_interacting_genes) for name in list(nx.nodes(network))}
        nx.set_node_attributes(network, interact_sars, 'interact_sars')
        nx.set_node_attributes(network, interact_sars_bool, 'interact_sars_bool')
    # Add log2FC between A549-ACE2virus vs. A549-ACE2novirus for terminals
    for network in networks_dict.values():
        log2FC = nx.get_node_attributes(network, 'log2FC_blanco')
        newlog2FC = {name: np.nan_to_num(log2FC[name]) for name in list(nx.nodes(network))}
        newabslog2FC = {name: np.abs(np.nan_to_num(log2FC[name])) for name in list(nx.nodes(network))}
        nx.set_node_attributes(network, newlog2FC, 'log2FCA549ACE2virusnovirus_all')
        nx.set_node_attributes(network, newabslog2FC, 'abslog2FCA549ACE2virusnovirus_all')
        
    return(networks_dict)


def make_summary(networks_dict, n_terminals, g=0):
    """ Create a summary of all networks obtained from the sensitivity analysis including informative
    metrics and statistics
    
    Args:
        networks_dict: dictionary of networks, output of add_metadata
        n_terminals: total number of terminal nodes
        g: parameter g used in the sensitivity analysis
    
    Returns:
        A dataframe including information on each augmented Steiner tree obtained in the sensitivity analysis
    """
    networks_summary = {}
    network_index = 0
    for paramstring, network in networks_dict.items():
        if network.number_of_nodes() != 0: 
            network_df = oi.get_networkx_graph_as_dataframe_of_nodes(network)
            networks_summary[paramstring] = {
                "index":                                network_index,
                "w":                                    paramstring.split("_")[1],
                "b":                                    paramstring.split("_")[3],
                "g":                                    g,
                "nodes":                                network.number_of_nodes(),
                "edges":                                network.number_of_edges(),
                "number_components":                    len([c for c in nx.connected_components(network)]),
                "size_components":                      [len(c) for c in sorted(nx.connected_components(network), key=len, reverse=True)],
                "percentage_terminals":                 np.sum(network_df['terminal'])/n_terminals,
                "upregulated_terminals":                network_df.loc[(network_df['terminal']==True) & (network_df['log2FCA549ACE2virusnovirus_all']>0)]['prize'].count(),
                "downregulated_terminals":              network_df.loc[(network_df['terminal']==True) & (network_df['log2FCA549ACE2virusnovirus_all']<0)]['prize'].count(),
                "number_transcription_regulators":      sum(network_df['general_function']=='transcription regulator'),
                "transcription_regulators":             list(network_df[network_df['general_function']=='transcription regulator'].index),
                "number_of_proteins_interacting_sars":  network_df.interact_sars_bool.sum(),
                "proteins_interacting_sars":            list(network_df[network_df['interact_sars_bool']==True].index)
            }
            network_index = network_index+1
    networks_summary_df = pd.DataFrame.from_dict(networks_summary, orient='index')
    return(networks_summary_df)


def create_matrix_gene_overlap_between_networks(networks_summary_df, networks_dict):
    """ Creates matrix where element (i,j) quantifies the number of common nodes in networks i and j
    divided by the total number of nodes in both networks
    
    Args:
        networks_summary_df: dataframe, output of make_summary
        networks_dict: dictionary of networks
    
    Returns:
        A matrix where element (i,j) quantifies the number of common nodes in networks i and j divided by the total number of nodes in both networks
    """
    N = len(networks_summary_df)
    mat_allnodes = np.zeros((N,N))
    for i in np.arange(N):
        for j in np.arange(i+1,N,1):
            # Select networks
            paramstring_a = networks_summary_df.loc[networks_summary_df['index']==i].index[0]
            paramstring_b = networks_summary_df.loc[networks_summary_df['index']==j].index[0]
            network_a = networks_dict[paramstring_a]
            network_b = networks_dict[paramstring_b]
            # Compute intersection/union
            intersection = float(len(set(network_a.nodes()).intersection(set(network_b.nodes()))))
            union = float(len(set(network_a.nodes()).union(set(network_b.nodes()))))
            mat_allnodes[i,j] = intersection/union    
    mat_allnodes = mat_allnodes+np.transpose(mat_allnodes)+np.diag(np.ones(N))
    
    return(mat_allnodes)

    
def create_matrix_terminal_overlap_between_networks(networks_summary_df, networks_dict):
    """ Creates matrix where element (i,j) quantifies the number of common terminals in networks i and j
    divided by the total number of terminals in both networks
    
    Args:
        networks_summary_df: dataframe, output of make_summary
        networks_dict: dictionary of networks
    
    Returns:
        A matrix where element (i,j) quantifies the number of common terminals in networks i and j divided by the total number of terminals in both networks
    """
    N = len(networks_summary_df)    
    mat_terminals = np.zeros((N,N))
    for i in np.arange(N):
        for j in np.arange(i+1,N,1):
            # Select networks
            paramstring_a = networks_summary_df.loc[networks_summary_df['index']==i].index[0]
            paramstring_b = networks_summary_df.loc[networks_summary_df['index']==j].index[0]
            network_a = networks_dict[paramstring_a]
            network_b = networks_dict[paramstring_b]
            # Compute intersection/union
            df_a = oi.get_networkx_graph_as_dataframe_of_nodes(network_a)
            terminals_in_a = set(df_a[df_a['terminal']==True].index)
            df_b = oi.get_networkx_graph_as_dataframe_of_nodes(network_b)
            terminals_in_b = set(df_b[df_b['terminal']==True].index)
            intersection = float(len(terminals_in_a.intersection(terminals_in_b)))
            union = float(len(terminals_in_a.union(terminals_in_b)))
            mat_terminals[i,j] = intersection/union    
    mat_terminals = mat_terminals+np.transpose(mat_terminals)+np.diag(np.ones(N))
    
    return(mat_terminals)


def create_matrix_sars_overlap_between_networks(networks_summary_df, networks_dict):
    """ Creates matrix where element (i,j) quantifies the number of common SARS-Cov-2 partners in networks i and j
    divided by the total number of SARS-Cov-2 partners in both networks
    
    Args:
        networks_summary_df: dataframe, output of make_summary
        networks_dict: dictionary of networks
    
    Returns:
        A matrix where element (i,j) quantifies the number of common SARS-Cov-2 partners in networks i and j divided by the total number of SARS-Cov-2 partners in both networks
    """
    N = len(networks_summary_df)
    mat_sars = np.zeros((N,N))
    for i in np.arange(N):
        for j in np.arange(i+1,N,1):
            # Select networks
            paramstring_a = networks_summary_df.loc[networks_summary_df['index']==i].index[0]
            paramstring_b = networks_summary_df.loc[networks_summary_df['index']==j].index[0]
            network_a = networks_dict[paramstring_a]
            network_b = networks_dict[paramstring_b]
            # Compute intersection/union
            df_a = oi.get_networkx_graph_as_dataframe_of_nodes(network_a)
            sars_in_a = set(df_a[df_a['interact_sars_bool']==True].index)
            df_b = oi.get_networkx_graph_as_dataframe_of_nodes(network_b)
            sars_in_b = set(df_b[df_b['interact_sars_bool']==True].index)
            intersection = float(len(sars_in_a.intersection(sars_in_b)))
            union = float(len(sars_in_a.union(sars_in_b)))
            mat_sars[i,j] = intersection/union    
    mat_sars = mat_sars+np.transpose(mat_sars)+np.diag(np.ones(N))
    
    return(mat_sars)
    
