# Causal Network Models of SARS-CoV-2 Expression and Aging to Identify Candidates for Drug Repurposing
<img align="center"  src="figure_method.png?raw=true">

This repository contains code for Causal Network Models of SARS-CoV-2 Expression and Aging to Identify Candidates for Drug Repurposing (https://arxiv.org/pdf/2006.03735.pdf) main methodology and analysis. 

Code_autoencoding directory contains code for training an over-parameterized autoencoder for mining relevant drugs for drug repurposing. Code_ppi directory contains scripts and Jupyter notebooks for running Steiner tree and causal analysis for investigating the drug mechanism and prioritizing drugs.

## System requirements
Ubuntu 16.04, 
Python 3.7, 
PyTorch 1.6 (CUDA enabled), 
cmapPy 4.0.1, 
[OmicsIntegrator 2](https://github.com/fraenkel-lab/OmicsIntegrator2), 
[causaldag 0.1a133](https://github.com/uhlerlab/causaldag), 
networkx (2.4), 
scikit-learn (0.22.2), 
graphviz (2.40.1)

## Installation guide
Clone this repository with the code (~5 secs):

```bash
git clone http://github.com/uhlerlab/covid19_repurposing.git
```

## Demo
For demos of autoencoder and protein-protein interaction analysis, see the READMEs in Code_autoencoding and Code_ppi, respectively. The expected outputs are shown in the accompanying Jupyter notebooks in [Code_ppi/Code/SteinerTree_notebook.ipynb](https://nbviewer.jupyter.org/github/uhlerlab/covid19_repurposing/blob/main/Code_ppi/Code/SteinerTree_notebook.ipynb) and [Code_ppi/Code/CausalAnalysis.ipynb](https://nbviewer.jupyter.org/github/uhlerlab/covid19_repurposing/blob/main/Code_ppi/Code/CausalAnalysis.ipynb). The expected runtime of the Jupyter notebooks is < 2 hours and the autoencoder training is < 2 hours.

## Instructions for use
In order to run the software on your data replace inputs in Jupyter notebooks in Code_ppi and scripts in Code_autoencoding with your own data.
