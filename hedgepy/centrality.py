import networkx as nx
import pandas as pd
import numpy as np

def average_degree_centrality(G):
    dc = nx.degree_centrality(G)
    avg_dc = np.mean(list(cbc.values()))
    return avg_dc

def average_eigenvector_centrality(G):
    ec = nx.eigenvector_centrality(G)
    avg_ec = np.mean(list(ec.values()))
    return avg_ec
