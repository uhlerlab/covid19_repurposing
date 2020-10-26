import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import graphviz
import matplotlib
import matplotlib.cm as cm


# define which genes are targets of drugs
rectangles = ['ACVR2A', 'AURKC', 'BRSK1', 'CDK17', 'EGFR', 'FGFR1', 'FGFR3', 'HDAC1', 'HSP90AA1', 'IRAK1', 'PAK1', 'PDE4B', 'RIPK1', 'RIPK2', 'STK3']


def plot_graphviz_style_graph(g, fname):
	agraph = nx.nx_agraph.to_agraph(g)
	agraph.graph_attr.update(size="21.26,35.08!")
	agraph.node_attr.update(style='filled', color = '#DCDCDC')
	# make certain nodes rectangles (targets of drugs)
	for node in g.nodes():
		if node in rectangles:
			n = agraph.get_node(node)
			n.attr['shape']='box'
	# color nodes according to log2 fold change of infection
	log2fc = pd.read_csv('../Save/blancoA_l2fc.csv', index_col=0)
	log2fc.index = [name.upper() for name in log2fc.index.values]
	log2fc_ppi = log2fc.loc[log2fc.index.intersection(list(g.nodes())), :]
	log2fc_ppi_pos = log2fc_ppi.loc[log2fc_ppi['0'] > 0, :]
	log2fc_ppi_neg = log2fc_ppi.loc[log2fc_ppi['0'] < 0, :]

	# map log2fc to colors
	update_node_colors(agraph, log2fc_ppi_pos, 'Reds', minima=0, maxima=3)
	update_node_colors(agraph, log2fc_ppi_neg, 'Blues', minima=-3, maxima=0)
	

	agraph.layout(prog='dot') # use dot
	agraph.draw(fname, format='png')
	return agraph


def update_node_colors(agraph, log2fc_ppi_pos, cmap, minima=None, maxima=None):
	lst = log2fc_ppi_pos.values.flatten()
	if minima is None and maxima is None:
		minima = min(lst)
		maxima = max(lst)

	norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
	mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

	for v, name in zip(lst, log2fc_ppi_pos.index.values):
		col = matplotlib.colors.rgb2hex(mapper.to_rgba(v))
		n = agraph.get_node(name)
		n.attr['color']=col
		n.attr['fillcolor']=col