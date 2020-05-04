import networkx as nx
import pandas as pd
import numpy as np

def average_degree_centrality(G):
    dc = nx.degree_centrality(G)
    avg_dc = np.mean(list(cbc.values()))
    return avg_dc
