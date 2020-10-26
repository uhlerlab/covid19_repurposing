#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:50:53 2020

@author: louis.cammarata
"""

# Imports
import numpy as np
import pandas as pd
import networkx as nx
import logging
logger = logging.getLogger(__name__)
import OmicsIntegrator as oi


def prepare_prizes_with_partners(graph, prize_file_name, virus_interacting_genes, terminal_partners,p):
    """ Add virus partners to teh terminal list with a prize p, without changing virus partners that are already part
    of the terminal list
    
    Args:
        graph: OmicsIntegrator2 graph on which the PCSF algorithm will be run
        prize_file: path to file containing terminal nodes and prizes
        virus_interacting_genes: set of SARS-Cov-2 interaction partners
        terminal_partners: set of SARS-Cov-2 interaction partners that are also terminal nodes
        p: prize given to SARS-Cov-2 interaction partners
    
    Returns:
        void
    """
    # Read original prize file
    prizes_dataframe = pd.read_csv(prize_file_name, sep='\t', na_filter=False)
    # Add virus partners
    for partner in list(virus_interacting_genes.difference(terminal_partners)):
        prizes_dataframe.loc[len(prizes_dataframe)] = [partner,p,0,0]
    # Prepare prizes
    prizes_dataframe.columns = ['name', 'prize'] + prizes_dataframe.columns[2:].tolist()
    prizes_dataframe['prize'] = pd.to_numeric(prizes_dataframe['prize'])
    graph._prepare_prizes(prizes_dataframe)
    
    
def run_prize_sensitivity_analysis(interactome_file_name, prize_file_name, graph_params, virus_interacting_genes, terminal_partners, P_list):
    """ Runs Steiner tree algorithm on the prized interactome with different values of the parameters p.
    Stores the results in a dictionary.
    
    Args:
        interactome_file_name: path to the IREF interactome
        prize_file_name: path to the terminal file (prized nodes)
        graph_params: dictionary of graph hyperparameters
        virus_interacting_genes: set of SARS-Cov-2 interaction partners
        terminal_partners: set of SARS-Cov-2 interaction partners that are also terminal nodes
        P_list: list of prizes given to SARS-Cov-2 interaction partners
    
    Returns:
        A dictionary of networks corresponding to different values of p.
    """
    # Build graph and define useful quantities
    graph = oi.Graph(interactome_file_name, graph_params)
    N = len(P_list)
    
    networks_dict = {}
    count = 0
    for i in np.arange(N):
        # Recompute prizes
        graph._reset_hyperparameters(graph_params)
        prepare_prizes_with_partners(graph, prize_file_name, virus_interacting_genes, terminal_partners,P_list[i])
        # Run PCSF
        vertex_indices, edge_indices = graph.pcsf()
        forest, augmented_forest = graph.output_forest_as_networkx(vertex_indices, edge_indices)
        # Store result in network
        key = "p_"+str(P_list[i])
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
    
    
def make_summary(networks_dict, n_terminals):
    """ Create a summary of all networks obtained from the sensitivity analysis including informative
    metrics and statistics
    
    Args:
        networks_dict: dictionary of networks, output of add_metadata
        n_terminals: total number of terminal nodes
    
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
                "p":                                    paramstring.split("_")[1],
                "nodes":                                network.number_of_nodes(),
                "edges":                                network.number_of_edges(),
                "number_components":                    len([c for c in nx.connected_components(network)]),
                "size_components":                      [len(c) for c in sorted(nx.connected_components(network), key=len, reverse=True)],
                "percentage_terminals":                 np.sum(network_df['abslog2FCA549ACE2virusnovirus_all']>0)/n_terminals,
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
        for j in np.arange(0,N,1):
            # Select networks
            paramstring_a = networks_summary_df.loc[networks_summary_df['index']==i].index[0]
            paramstring_b = networks_summary_df.loc[networks_summary_df['index']==j].index[0]
            network_a = networks_dict[paramstring_a]
            network_b = networks_dict[paramstring_b]
            # Compute difference/number of nodes in a
            nodes_in_a = set(network_a.nodes())
            nodes_in_b = set(network_b.nodes())
            difference = float(len(nodes_in_a.difference(nodes_in_b)))
            mat_allnodes[i,j] = difference/float(len(nodes_in_a))    
    
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
            terminals_in_a = set(df_a[df_a['abslog2FCA549ACE2virusnovirus_all']>0].index)
            df_b = oi.get_networkx_graph_as_dataframe_of_nodes(network_b)
            terminals_in_b = set(df_b[df_b['abslog2FCA549ACE2virusnovirus_all']>0].index)
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
    mat_sars = np.zeros((N,N))
    for i in np.arange(N):
        for j in np.arange(0,N,1):
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
            difference = float(len(sars_in_a.difference(sars_in_b)))
            mat_sars[i,j] = difference/float(len(sars_in_a))
    
    return(mat_sars)



















