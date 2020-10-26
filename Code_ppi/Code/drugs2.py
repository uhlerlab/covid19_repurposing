#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 16:15:28 2020

@author: louis.cammarata
"""

# Import
import numpy as np
import pandas as pd
import networkx as nx
import logging
logger = logging.getLogger(__name__)
import OmicsIntegrator as oi


def unnesting(df, explode):
    """ Helper function to explode a dataframe based on a given column
    
    Args:
        df: the datframe to explode
        explode = list of columns to explode on
    
    Returns:
        Exploded dataframe
    """
    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx
    return df1.join(df.drop(explode, 1), how='left')


def load_drug_target_data(drug_target_file_name, aff_cst_thresh=5):
    """ Loads and processes drug/target data from DrugCentral
    
    Args:
        drug_target_file_name: path to DrugCentral data set
        aff_cst_thresh: upper threshold on -log(affinity_constant)
    
    Returns:
        A dataframe of genes, compounds targetting these genes and the corresponding affinity constant
    """
    # Load data
    drugcentral_df = pd.read_csv(drug_target_file_name, sep='\t')
    
    # Only keep Homo Sapiens gene targets
    drugcentral_df = drugcentral_df.loc[drugcentral_df['ORGANISM']=='Homo sapiens']
    
    # Keep relevant columns only
    drugcentral_df = drugcentral_df[['DRUG_NAME','TARGET_NAME','GENE','ACT_VALUE','ACT_TYPE']]
    
    # Turn gene column into list of strings
    drugcentral_df['GENE'] = drugcentral_df['GENE'].str.split('|')
    
    # Explode dataframe based on gene names
    drugcentral_df = unnesting(drugcentral_df, explode = ['GENE'])
    
    # Rename columns
    drugcentral_df.columns = ['gene','drug','protein_target','affinity_constant','affinity_constant_type']
    
    # Minusculize all drug names for standardization
    drugcentral_df['drug'] = drugcentral_df['drug'].str.lower().str.strip()
    
    # Threshold affinity constants (the value we have is -log(constant), so to have constant<10muM we need -log(constant)>5)
    mask = drugcentral_df['affinity_constant'] < aff_cst_thresh
    drugcentral_df = drugcentral_df.loc[~mask]
    
    return(drugcentral_df)


def load_embedded_drugs(embedded_drugs_file_name, lower_corr_thresh=-2):
    """ Loads and processes drug/target data from DrugCentral
    
    Args:
        drug_target_file_name: path to DrugCentral data set
        aff_cst_thresh: upper threshold on -log(affinity_constant)
    
    Returns:
        A dataframe of genes, compounds targetting these genes and the corresponding affinity constant
    """

    # Load data
    bestdrugs_df =  pd.read_csv(embedded_drugs_file_name, sep=',',header=0, names = ['drug','corr','original_corr','pca_corr'], dtype = str)
    
    # Clean file
    bestdrugs_df['drug'] = bestdrugs_df['drug'].str.lower().str.strip()
    bestdrugs_df['corr'] = pd.to_numeric(bestdrugs_df['corr'], downcast='float')
    bestdrugs_df['original_corr'] = pd.to_numeric(bestdrugs_df['original_corr'], downcast='float')
    bestdrugs_df['pca_corr'] = pd.to_numeric(bestdrugs_df['pca_corr'], downcast='float')
    
    
    # Threshold low correlation drugs (also see lower_corr_thresh=0.86)
    bestdrugs_df = bestdrugs_df.loc[bestdrugs_df['corr']>lower_corr_thresh][['drug','corr']]
    
    return(bestdrugs_df)


def add_drug_info_to_selected_network(network_selected, targets_and_drugs_df):
    """ Adds drug information to the currently selected network
    
    Args:
        network_selected: NetworkX network
        targets_and_drugs_df: dataframe of drugs and their target genes
    
    Returns:
        NetworkX network with drug information
    """   
    # Add following attribute for each gene: is gene druggable, what drug 
    drugs = {gene:list(targets_and_drugs_df.loc[targets_and_drugs_df['gene']==gene]['drug']) for gene in list(network_selected.nodes())}
    drugs_with_corr_aff = {gene:[targets_and_drugs_df['drug'][i]+'%'+str(targets_and_drugs_df['corr'][i])+'%'+str(targets_and_drugs_df['affinity_constant'][i])
                             for i in targets_and_drugs_df.index[targets_and_drugs_df['gene']==gene]]
                             for gene in list(network_selected.nodes())}
    druggable_boolean = {gene:drugs[gene]!=[] for gene in list(network_selected.nodes())}
    protein_target = {gene:list(targets_and_drugs_df.loc[targets_and_drugs_df['gene']==gene]['protein_target']) for gene in list(network_selected.nodes())}
    nx.set_node_attributes(network_selected, druggable_boolean, name='druggable')
    nx.set_node_attributes(network_selected, drugs, name='drug')
    nx.set_node_attributes(network_selected, drugs_with_corr_aff, name='drug_with_corr_aff')
    nx.set_node_attributes(network_selected, protein_target, name='protein_target')
    
    return(network_selected)


def drug_targets_in_selected_network(network_selected):
    """ Creates table of drug targets in selected network along with relevant metadata
    
    Args:
        network_selected: NetworkX network
    
    Returns:
        Dataframe of drug targets in selected network along woth relevant metadata
    """   
    # Get selected network as dataframe of node
    network_enriched_df = oi.get_networkx_graph_as_dataframe_of_nodes(network_selected)
    
    # Drug targets in network dataframe
    drug_targets_df = network_enriched_df.copy()
    drug_targets_df = drug_targets_df.loc[drug_targets_df['druggable']==True]
    drug_targets_df['name'] = drug_targets_df.index
    wanted_columns = ['name','drug_with_corr_aff','protein_target']
    drug_targets_df = drug_targets_df[wanted_columns]
    drug_targets_df['protein_target'] = [set(drug_targets_df['protein_target'][i]) for i in np.arange(len(drug_targets_df))]
    
    # Explode
    drug_targets_df = unnesting(drug_targets_df, ['drug_with_corr_aff'])
    drug_targets_df[['drug','corr','affinity']] = drug_targets_df['drug_with_corr_aff'].str.split('%',expand = True)
    wanted_columns2 = ['name','protein_target','drug','corr','affinity']
    drug_targets_df = drug_targets_df[wanted_columns2]
    
    return(drug_targets_df)
