#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 16:47:02 2020

@author: louis.cammarata
"""

# Imports
import pandas as pd
import numpy as np
from matplotlib_venn import venn2, venn3
import matplotlib.pyplot as plt
import OmicsIntegrator as oi

def load_protein_coding_genes(protein_coding_genes_file_name):
    """ Loads and processes list of protein coding genes

    Args:
        protein_coding_genes_file_name: path to protein coding genes data set

    Returns:
        A dataframe of protein coding genes

    """

    coding_genes = pd.read_csv(protein_coding_genes_file_name, sep = ' ')
    coding_genes = coding_genes.loc[:,['hgnc_symbol']].dropna()
    coding_genes = coding_genes.rename(columns = {'hgnc_symbol': 'name'})
    coding_genes['name'] = coding_genes['name'].str.upper().str.strip()
    coding_genes = coding_genes[~coding_genes.duplicated(subset = ['name'], keep='first')]

    return(coding_genes)


def load_iref_genes(ppi_data_file_name):
    """ Loads and processes IREF protein-protein interaction data
    
    Args:
        ppi_data_file_name: path to ppi data set
    
    Returns:
        A dataframe of proteins present in the IREF interactome
    
    """
    
    ppi_net = oi.Graph(ppi_data_file_name)
    proteins_in_ppi_df = pd.DataFrame(data = {'name': list(ppi_net.nodes)})
    proteins_in_ppi = proteins_in_ppi_df['name'].str.upper().str.strip()
    
    return(proteins_in_ppi)


def load_lincs_genes(lincs_genes_file_name):
    """ Loads and processes genes in LINCS
    
    Args:
        lincs_genes_file_name: path to lincs genes info data set
    
    Returns:
        A list of genes present in LINCS
    
    """

    lincs_df = pd.read_csv(lincs_genes_file_name, sep='\t')
    l1000_genes = list(set(lincs_df[lincs_df['pr_is_lm'] == 1]['pr_gene_symbol']))
    
    return(l1000_genes)


def load_selected_aging_rnaseq_data(aging_rnaseq_data_file_name,
                                   metadata_file_name,
                                   protein_coding_genes_file_name):
    """ Performs differential expression analysis on aging genes from GTEX. Samples are chosen
    from a young group (20-29 years old) and from an old group (70-79 years old). This function loads the 
    data, performs mundane processing, computes log2FC between old and young. Only genes that are sufficiently
    expressed in both conditions are considered. We select the top 0.1*N_coding genes from the list of genes 
    ranked in desceding absolute log2FC order.
    
    Args:
        aging_rnaseq_data_file_name: path to aging rnaseq data set
        metadata_file_name: path to metadata on the aging rnaseq data set
        protein_coding_genes_file_name: path to protein coding genes data set

    Returns:
        A dataframe of selected age-related genes
    """

    # Load aging RNAseq data
    gtex_rpkm0 = pd.read_csv(aging_rnaseq_data_file_name, index_col=0)
    gtex_rpkm0.columns = [col.replace('.','-') for col in gtex_rpkm0.columns.values]
    
    # Load metadata and select young columns (20-29 y.o.) and old columns (70-79 y.o.)
    metadata_aging = pd.read_csv(metadata_file_name, index_col=0)
    young_cols = metadata_aging.loc[metadata_aging['AGE'] == '20-29', :].index.values
    old_cols = metadata_aging.loc[metadata_aging['AGE'] == '70-79', :].index.values
    
    # Remove inf and nan from aging RNAseq data
    gtex_rpkm = gtex_rpkm0.replace([np.inf, -np.inf], np.nan)
    gtex_rpkm = gtex_rpkm.dropna(subset=np.concatenate((young_cols, old_cols)), how='any')
    
    # Change data from log2(1+RPKM) to RPKM
    for col in gtex_rpkm.columns:
        gtex_rpkm[col] = 2**gtex_rpkm[col]-1
    gtex_rpkm['name'] = gtex_rpkm.index.str.upper().str.strip()
    
    # Compute mean in treatment (old) vs. control (young)
    gtex_rpkm['mean_treatment'] = gtex_rpkm[old_cols].mean(axis = 1)
    gtex_rpkm['mean_control'] = gtex_rpkm[young_cols].mean(axis = 1)
    gtex_rpkm = gtex_rpkm[['name','mean_treatment','mean_control']]

    # Keep only protein coding
    coding_genes = load_protein_coding_genes(protein_coding_genes_file_name)
    gtex_rpkm = gtex_rpkm.merge(coding_genes, on = 'name', how = 'inner')

    # Add log2FC and absolute log2FC
    gtex_rpkm['log2FC'] = np.log2(1+gtex_rpkm['mean_treatment'])-np.log2(1+gtex_rpkm['mean_control'])
    gtex_rpkm['abslog2FC'] = np.abs(gtex_rpkm['log2FC'])

    # Drop gene in lower square (weak expression in both control and treatment)
    tlow = 1
    mask = (np.log2(1+gtex_rpkm['mean_treatment'])<tlow) & (np.log2(1+gtex_rpkm['mean_control'])<tlow)
    gtex_rpkm = gtex_rpkm[~mask]

    # Select top 10% genes
    N_coding = len(coding_genes)
    number_selected = int(0.1*N_coding)
    gtex_rpkm = gtex_rpkm.sort_values(['abslog2FC'], ascending=False)
    gtex_rpkm_selected_df = gtex_rpkm[0:number_selected]

    return(gtex_rpkm_selected_df)


def load_and_process_blanco_data(blanco_data_file_name,
                                 protein_coding_genes_file_name,
                                 shuffle=False,
                                 seed=13):
    """ Loads and pre-processes rnaseq data from Blanco et al. Buils three data frames with two groups:
        * blanco1_rpkm contains data for A549 cells without SARS-Cov-2 (control) and with SARS-Cov-2 (treatment)
        * blanco2_rpkm contains data for A549-ACE2 cells without SARS-Cov-2 (control) and with SARS-Cov-2 (treatment)
        * blanco3_rpkm contains data for A549 cells without ACE2 (control) and with ACE2 (treatment)
    
    Args:
        blanco_data_file_name: path to blanco rnaseq data set
        protein_coding_genes_file_name: path to protein coding genes data set
        shuffle: shether to shuffle gene names in the Blanco et al. data to perform specificity analysis
        seed: random seed if shuffle=True

    Returns:
        Three data frames of correspondong to blanco1_rpkm, blanco2_rpkm and blanco3_rpkm described above
    """

    # Load blanco data and select relevant series
    blanco_rpkm = pd.read_csv(blanco_data_file_name, index_col=0)
    blanco_rpkm = blanco_rpkm.replace([np.inf, -np.inf], np.nan)
    blanco_rpkm = blanco_rpkm.dropna(subset = ['Series5_A549_Mock_1',
                                               'Series5_A549_Mock_2',
                                               'Series5_A549_Mock_3',
                                               'Series5_A549_SARS.CoV.2_1',
                                               'Series5_A549_SARS.CoV.2_2',
                                               'Series5_A549_SARS.CoV.2_3',
                                               'Series16_A549.ACE2_Mock_1',
                                               'Series16_A549.ACE2_Mock_2',
                                               'Series16_A549.ACE2_Mock_3',
                                               'Series16_A549.ACE2_SARS.CoV.2_1',
                                               'Series16_A549.ACE2_SARS.CoV.2_2',
                                               'Series16_A549.ACE2_SARS.CoV.2_3'],
                                        how = 'any')
    # Shuffle the data if shuffle=True
    if shuffle == True:
        np.random.seed(seed)
        blanco_rpkm.index = np.random.choice(list(blanco_rpkm.index), size = len(blanco_rpkm.index), replace=False) 
    
    # Change data from log2(1+RPKM) to RPKM
    for col in blanco_rpkm.columns:
        blanco_rpkm[col] = 2**blanco_rpkm[col]-1
    blanco_rpkm['name'] = blanco_rpkm.index.str.upper().str.strip()

    # Construct blanco1 (A549 Mock vs. SARS-Cov-2), blanco2 (A549-ACE2 Mock vs. SARS-Cov-2), blanco3 (A549 Mock vs. A549-ACE2 Mock)
    blanco1_columns = ['name','Series5_A549_Mock_1', 'Series5_A549_Mock_2', 'Series5_A549_Mock_3',
                       'Series5_A549_SARS.CoV.2_1','Series5_A549_SARS.CoV.2_2', 'Series5_A549_SARS.CoV.2_3']
    blanco2_columns = ['name','Series16_A549.ACE2_Mock_1', 'Series16_A549.ACE2_Mock_2', 'Series16_A549.ACE2_Mock_3',
                       'Series16_A549.ACE2_SARS.CoV.2_1','Series16_A549.ACE2_SARS.CoV.2_2', 'Series16_A549.ACE2_SARS.CoV.2_3']
    blanco3_columns = ['name','Series16_A549.ACE2_Mock_1', 'Series16_A549.ACE2_Mock_2', 'Series16_A549.ACE2_Mock_3',
                       'Series5_A549_Mock_1', 'Series5_A549_Mock_2', 'Series5_A549_Mock_3']
    blanco1_rpkm = blanco_rpkm.copy()[blanco1_columns]
    blanco2_rpkm = blanco_rpkm.copy()[blanco2_columns]
    blanco3_rpkm = blanco_rpkm.copy()[blanco3_columns]

    # Create mean_treatment and mean_control columns
    blanco1_rpkm['mean_treatment'] = blanco1_rpkm.filter(regex = 'Series5_A549_SARS.CoV.2').mean(axis = 1)
    blanco1_rpkm['mean_control'] = blanco1_rpkm.filter(regex = 'Series5_A549_Mock').mean(axis = 1)
    blanco2_rpkm['mean_treatment'] = blanco2_rpkm.filter(regex = 'Series16_A549.ACE2_SARS.CoV.2').mean(axis = 1)
    blanco2_rpkm['mean_control'] = blanco2_rpkm.filter(regex = 'Series16_A549.ACE2_Mock').mean(axis = 1)
    blanco3_rpkm['mean_treatment'] = blanco3_rpkm.filter(regex = 'Series16_A549.ACE2_Mock').mean(axis = 1)
    blanco3_rpkm['mean_control'] = blanco3_rpkm.filter(regex = 'Series5_A549_Mock').mean(axis = 1)

    # Drop gene in lower square for all conditions (genes that are lowly expressed for all conditions and all comparisons)
    tlow = 1
    mask = (np.log2(1+blanco1_rpkm['mean_treatment'])<tlow) & (np.log2(1+blanco1_rpkm['mean_control'])<tlow) & (np.log2(1+blanco2_rpkm['mean_treatment'])<tlow) & (np.log2(1+blanco2_rpkm['mean_control'])<tlow) &(np.log2(1+blanco3_rpkm['mean_treatment'])<tlow) & (np.log2(1+blanco3_rpkm['mean_control'])<tlow)
    blanco1_rpkm = blanco1_rpkm[~mask]
    blanco2_rpkm = blanco2_rpkm[~mask]
    blanco3_rpkm = blanco3_rpkm[~mask]

    # Create log2FC treatment vs. control colum
    blanco1_rpkm['log2FC'] = np.log2(1+blanco1_rpkm['mean_treatment'])-np.log2(1+blanco1_rpkm['mean_control'])
    blanco2_rpkm['log2FC'] = np.log2(1+blanco2_rpkm['mean_treatment'])-np.log2(1+blanco2_rpkm['mean_control'])
    blanco3_rpkm['log2FC'] = np.log2(1+blanco3_rpkm['mean_treatment'])-np.log2(1+blanco3_rpkm['mean_control'])

    # Add abslog2FC column
    blanco1_rpkm['abslog2FC'] = np.abs(blanco1_rpkm['log2FC'])
    blanco2_rpkm['abslog2FC'] = np.abs(blanco2_rpkm['log2FC'])
    blanco3_rpkm['abslog2FC'] = np.abs(blanco3_rpkm['log2FC'])

    # Keep only protein coding genes
    coding_genes = load_protein_coding_genes(protein_coding_genes_file_name)
    blanco1_rpkm = blanco1_rpkm.merge(coding_genes, on = 'name', how = 'inner')
    blanco2_rpkm = blanco2_rpkm.merge(coding_genes, on = 'name', how = 'inner')
    blanco3_rpkm = blanco3_rpkm.merge(coding_genes, on = 'name', how = 'inner')

    return blanco1_rpkm, blanco2_rpkm, blanco3_rpkm


def load_selected_blanco_genes(blanco1_rpkm,
                               blanco2_rpkm,
                               blanco3_rpkm,
                               protein_coding_genes_file_name,
                               plot_venn_diagrams=True):
    """ Loads genes selected from blanco data set
    
    Args:
        blanco1_rpkm: data for A549 cells without SARS-Cov-2 (control) and with SARS-Cov-2 (treatment)
        blanco2_rpkm: contains data for A549-ACE2 cells without SARS-Cov-2 (control) and with SARS-Cov-2 (treatment)
        blanco3_rpkm: contains data for A549 cells without ACE2 (control) and with ACE2 (treatment)
        protein_coding_genes_file_name: path to protein coding genes data set
        plot_venn_diagrams: whether to plot a venn diagram for sets A, B and C

    Returns:
        Dataframe of selected genes from Blanco et al along with log2FC in blanco2_rpkm
    """
    
    # Assess number of selected genes in A\(BUC) for different thresholds on the absolute log2FC
    selected_genes_vec = []
    m = np.nanmax(np.concatenate((blanco1_rpkm['abslog2FC'], 
                                  blanco2_rpkm['abslog2FC'],
                                  blanco3_rpkm['abslog2FC']), axis=None))
    trange = np.arange(0,m, 0.001)
    for t in trange:
        A = blanco2_rpkm[blanco2_rpkm['abslog2FC']>t]['name']
        B = blanco1_rpkm[blanco1_rpkm['abslog2FC']>t]['name']
        C = blanco3_rpkm[blanco3_rpkm['abslog2FC']>t]['name']
        selected_genes_vec.append(len(set(A).difference(set(B).union(set(C)))))

    # For threshold pick value corresponding to the first index after which the number of selected genes is less than SIZE
    coding_genes = load_protein_coding_genes(protein_coding_genes_file_name)
    N_genes = len(coding_genes)
    number_selected = int(0.1*N_genes)
    SIZE = number_selected
    threshold_2 = trange[np.max(np.where(np.array(selected_genes_vec)>=SIZE))]

    # Create subset A: log_2 fold change of A459+virus+ACE2 divided by A459-virus+ACE2 in absolute value and obtain the subset of genes with above threshold value
    A = blanco2_rpkm[blanco2_rpkm['abslog2FC']>threshold_2]['name']
    # Create subset B: log_2 fold change of A459+virus-ACE2 divided by A459-virus-ACE2 in absolute value and obtain the subset of genes with above threshold value
    B = blanco1_rpkm[blanco1_rpkm['abslog2FC']>threshold_2]['name']
    # Create subset C: log_2 fold change of A459-virus+ACE2 divided by A459-virus-ACE2 in absolute value and obtain the subset of genes with above threshold value
    C = blanco3_rpkm[blanco3_rpkm['abslog2FC']>threshold_2]['name']

    if plot_venn_diagrams == True:
        venn3(subsets = [set(A),set(B),set(C)], set_labels = ('A','B','C'))

    # Final selected genes in A\(BUC)
    blanco_rpkm_genes_selected = list(set(A).difference(set(B).union(set(C))))
    blanco_rpkm_selected = blanco2_rpkm[[name in blanco_rpkm_genes_selected for name in blanco2_rpkm['name']]]

    return(blanco_rpkm_selected)


def create_prized_genes_list(blanco2_rpkm,
                             blanco_rpkm_selected,
                             gtex_rpkm_selected_df,
                             l1000_genes,
                             proteins_in_ppi,
                             plot_venn_diagrams=True):
    """ Creates dataframe of prized genes for downstream Steiner tree analysis
    
    Args:
        blanco2_rpkm: contains data for A549-ACE2 cells without SARS-Cov-2 (control) and with SARS-Cov-2 (treatment)
        blanco_rpkm_selected: dataframe of selected genes from Blanco et al along with log2FC in blanco2_rpkm
        gtex_rpkm_selected_df: dataframe of selected age-related genes
        l1000_genes: list of genes present in LINCS
        proteins_in_ppi: dataframe of protein coding genes
        plot_venn_diagrams: whether to plot a venn diagram for sets A, B and C

    Returns:
        Dataframe of selected genes from Blanco et al along with log2FC in blanco2_rpkm
    """
    
    # Define set of genes selected from Blanco et al.
    blanco_rpkm_selected_genes = set(blanco_rpkm_selected['name'])
    # Define set of genes selected from aging data
    gtex_rpkm_selected_genes = set(gtex_rpkm_selected_df['name'])
    # Define set of genes in LINCS
    lincs_genes = set(l1000_genes)

    if plot_venn_diagrams == True:
        venn3(subsets = [blanco_rpkm_selected_genes,gtex_rpkm_selected_genes,lincs_genes], set_labels = ('blanco','age','lincs'))
        plt.title('Intersection of selected blanco genes, selected aging genes and lincs genes')
        plt.show()
        venn2(subsets = [blanco_rpkm_selected_genes,gtex_rpkm_selected_genes],set_labels = ('blanco','age'))
        plt.title('Intersection of selected blanco genes and selected aging genes')
        plt.show()

    # Simplify set of selected ageing genes with ageing log2FC information
    gtex_rpkm_selected_df_select_columns = gtex_rpkm_selected_df.loc[:,['name', 'log2FC']]
    gtex_rpkm_selected_df_select_columns.columns = ['name', 'log2FC_ageing']
    gtex_rpkm_selected_df_select_columns.sort_values(by='name', inplace=True)
    gtex_rpkm_selected_df_select_columns.head()

    # Build final dataframe for PPI analysis with blanco data
    terminal_df = blanco2_rpkm.loc[blanco2_rpkm['name'].isin(blanco_rpkm_selected_genes), ['name','log2FC']]
    terminal_df.columns = ['name','log2FC_blanco']
    terminal_df.sort_values(by='name', inplace=True)

    # Merge terminal_df with ageing dataset
    terminal_df = terminal_df.merge(gtex_rpkm_selected_df_select_columns, on = 'name', how = 'inner')
    terminal_df.head()

    # Only keep genes that have same log2FC sign in blanco and ageing
    terminal_df = terminal_df.loc[terminal_df['log2FC_blanco']*terminal_df['log2FC_ageing']>=0]

    if plot_venn_diagrams == True:
        venn3(subsets = [blanco_rpkm_selected_genes,gtex_rpkm_selected_genes,set(terminal_df['name'])],set_labels = ('SARS-Cov-2','Age',''))
        plt.title('Intersection of selected consistent blanco genes and selected aging genes')
        plt.show()

    # Only keep genes present in the PPI
    terminal_df = terminal_df.loc[terminal_df['name'].isin(proteins_in_ppi)]

    # Add prize (abslog2FC from blanco)
    terminal_df.insert(1,'prize',np.abs(terminal_df['log2FC_blanco']))

    return(terminal_df)


def create_random_prized_genes_list( blanco2_rpkm,
                                     proteins_in_ppi,
                                     n_terminals,
                                     seed=13):
    """ Creates dataframe of random prized genes for downstream Steiner tree analysis
    
    Args:
        blanco2_rpkm: contains data for A549-ACE2 cells without SARS-Cov-2 (control) and with SARS-Cov-2 (treatment)
        l1000_genes: list of genes present in LINCS
        proteins_in_ppi: dataframe of protein coding genes
        n_terminals: number of terminal genes to select
        seed: random seed for reproducibility

    Returns:
        Dataframe of (randomly) selected genes from Blanco et al along with log2FC in blanco2_rpkm
    """
    
    # Select n_terminals genes at random from blanco2_rpkm
    np.random.seed(seed)
    terminal_df = blanco2_rpkm.iloc[np.random.choice(range(len(blanco2_rpkm)), size = n_terminals, replace=False)]
    terminal_df = terminal_df[['name','log2FC']]
    terminal_df.columns = ['name','log2FC_blanco']

    # Only keep genes present in the PPI
    terminal_df = terminal_df.loc[terminal_df['name'].isin(proteins_in_ppi)]

    # Add prize (abslog2FC from blanco)
    terminal_df.insert(1,'prize',np.abs(terminal_df['log2FC_blanco']))
    
    # Ad dummy log2FC_ageing column for compatibility with subsequent functions
    terminal_df['log2FC_ageing'] = 0

    return(terminal_df)
