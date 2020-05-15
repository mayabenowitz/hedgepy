import networkx as nx
import pandas as pd
import numpy as np

def average_degree_centrality(G):
    dc = nx.degree_centrality(G)
    avg_dc = np.mean(list(dc.values()))
    return avg_dc

def average_eigenvector_centrality(G):
    ec = nx.eigenvector_centrality(G, weight='weight')
    avg_ec = np.mean(list(ec.values()))
    return avg_ec

def average_closeness_centrality(G):
    cc = nx.closeness_centrality(G,)
    avg_cc = np.mean(list(cc.values()))
    return avg_cc

def average_communicability_centrality(G):
    cbc = nx.communicability_betweenness_centrality(G)
    avg_cbc = np.mean(list(cbc.values()))
    return avg_cbc

def global_communicability(nx_ts):
    nx_time_series = {
        timestamp: average_communicability_centrality(G)
        for timestamp, G in nx_ts.items()
    }

    df = pd.DataFrame.from_dict(nx_time_series, orient='index')\
        .rename(columns={0: 'global_communicability'})
    return df

def global_eigencentrality(nx_ts):
    nx_time_series = {
        timestamp: average_eigenvector_centrality(G)
        for timestamp, G in nx_ts.items()
    }

    df = pd.DataFrame.from_dict(nx_time_series, orient='index')\
        .rename(columns={0: 'global_eigencentrality'})
    return df

def global_degree_centrality(nx_ts):
    nx_time_series = {
        timestamp: average_degree_centrality(G)
        for timestamp, G in nx_ts.items()
    }

    df = pd.DataFrame.from_dict(nx_time_series, orient='index')\
        .rename(columns={0: 'global_degree_centrality'})
    return df

def global_closeness(nx_ts):
    nx_time_series = {
        timestamp: average_closeness_centrality(G)
        for timestamp, G in nx_ts.items()
    }

    df = pd.DataFrame.from_dict(nx_time_series, orient='index')\
        .rename(columns={0: 'global_communicability'})
    return df

def global_clustering(nx_ts):
    nx_time_series = {
        timestamp: nx.average_clustering(G, weight='weight')
        for timestamp, G in nx_ts.items()
    }

    df = pd.DataFrame.from_dict(nx_time_series, orient='index')\
        .rename(columns={0: 'global_clustering'})
    return df
