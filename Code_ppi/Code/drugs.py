#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:12:45 2020

@author: louis.cammarata
"""

# Imports
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)


def preprocess_drug_target_data(drug_target_file_name):
    """ Preprocesses the drug/target data from DrugCentral to obtain a dataframe of drugs and their corresponding targets
    
    Args:
        drug_target_file_name: path to the drug/target data
            
    Returns:
        A pandas dataframe with two columns, drug and targets
    
    
    """ 
    # Load drug/target data
    protcompound_full_df = pd.read_csv(drug_target_file_name, sep=',')
    
    # Drop duplicates and remove rows with missing values
    protcompound_full_df = protcompound_full_df.drop_duplicates(subset=None, keep="first", inplace=False)
    protcompound_full_df = protcompound_full_df.dropna()
    
    # Only keep drug and target columns
    protcompound_df = protcompound_full_df[['pert_iname','Gene_Name']]
    protcompound_df.columns = ['drug','target']
    
    # Group by drugs and aggregate corresponding targets into a string
    protcompound_df_merged = protcompound_df.groupby('drug', as_index = False).agg({'target':','.join})
    protcompound_df_simple = protcompound_df_merged
    protcompound_df_simple['target'] = [set(protcompound_df_merged['target'][i].split(',')).difference(set({''})) for i in np.arange(len(protcompound_df_simple))]
    
    return(protcompound_df_simple)


def drugs_from_embedding(drugs_from_embedding_file_name):
    """ Preprocesses the selected drugs after embedding
    
    Args:
        drugs_from_embedding_file_name: path to the selected drugs data using the embedding
        
    Returns:
        A pandas dataframe with two columns, drug and anti-correlation with the SARS-Cov-2 signature in the embedded space
    
    """
    # Load drugs from ebeddign data
    bestdrugs_df =  pd.read_csv(drugs_from_embedding_file_name, sep=',',header=None, names = ['drug','corr'])
    # Format data
    bestdrugs_df['drug'] = bestdrugs_df['drug'].str.split('\'',expand = True)[1].str.lower().str.strip()
    bestdrugs_df['corr'] = bestdrugs_df['corr'].str.split(')',expand = True)[0]
    
    return(bestdrugs_df)


def drugs_from_embedding_with_targets(drugs_from_embedding, drugs_with_targets):
    """ Creates a dataframe where each selected drug (drugs that are anticorrelated with Sars-Cov-2 signature in the embedded space)
        is mapped to its corresponding targets
        
    
    Args:
        drugs_from_embedding: pandas dataframe, output of drugs_from_embedding()
        drugs_with_targets: pandas dataframe, ouput of preprocess_drug_target_data()
        
    Returns:
        A pandas dataframe with three columns: drug, anti-correlation with the SARS-Cov-2 signature in the embedded space, targets
    
    """
    # Perform inner join of drugs_from_embedding and drugs_with_targets
    bestdrugs_to_proteins_df = drugs_from_embedding.merge(drugs_with_targets, on = 'drug', how = 'inner')
    
    return(bestdrugs_to_proteins_df)

