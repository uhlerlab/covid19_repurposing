#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 16:47:02 2020

@author: louis.cammarata
"""

# Imports
import prizes

# Test load_protein_coding_genes
protein_coding_genes_file_name = '../Data/protein_coding_ensembl_gene_id_hgnc_hg19.txt'
protein_coding_genes = prizes.load_protein_coding_genes(protein_coding_genes_file_name)
print(protein_coding_genes.head())

# Test load_iref_genes
ppi_data_file_name = '../Data/iRefIndex_v14_MIScore_interactome_C9.costs.txt'
proteins_in_ppi = prizes.load_iref_genes(ppi_data_file_name)
print(proteins_in_ppi.head())

# Test load_lincs_genes
lincs_genes_file_name = '../Data/GSE92742_Broad_LINCS_gene_info.txt'
l1000_genes = prizes.load_lincs_genes(lincs_genes_file_name)
print(len(l1000_genes))

# Test load_selected_aging_rnaseq_data
aging_rnaseq_data_file_name = '../Data/GTEX_log2_RPKMquantile.csv'
metadata_file_name = '../Data/metadata.csv.gz'
protein_coding_genes_file_name = '../Data/protein_coding_ensembl_gene_id_hgnc_hg19.txt'
gtex_rpkm_selected_df = prizes.load_selected_aging_rnaseq_data(aging_rnaseq_data_file_name,metadata_file_name, protein_coding_genes_file_name)
print(gtex_rpkm_selected_df.head())

# Test load_and_process_blanco_data
blanco_data_file_name = '../Data/GSE147507_log2_RPKMquantile.csv'
blanco1_rpkm, blanco2_rpkm, blanco3_rpkm = prizes.load_and_process_blanco_data(blanco_data_file_name, protein_coding_genes_file_name)
print(blanco2_rpkm.head())

# Test load_selected_blanco_genes
blanco_rpkm_selected = prizes.load_selected_blanco_genes( blanco1_rpkm,
                                                   blanco2_rpkm,
                                                   blanco3_rpkm,
                                                   protein_coding_genes_file_name,
                                                   plot_venn_diagrams=False)
print(blanco_rpkm_selected.head())

# Test create_prized_genes_list
terminal_df = prizes.create_prized_genes_list(blanco2_rpkm,
                                     blanco_rpkm_selected,
                                     gtex_rpkm_selected_df,
                                     l1000_genes,
                                     proteins_in_ppi,
                                     plot_venn_diagrams=False)
print(terminal_df.head())
