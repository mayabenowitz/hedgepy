from typing import Dict, List
import pandas as pd
import dcor
import networkx as nx
from memoization import cached


def ticker_time_frame(df: pd.DataFrame, ticker_col_name: str) -> Dict[str, pd.DataFrame]:
    """Returns a dictionary of pivoted dataframes"""

    try:
        assert isinstance(df.index, pd.DatetimeIndex)
    except AssertionError:
        print("Please pass a DataFrame with a DatetimeIndex")

    time_series = [col for col in df.columns.tolist() if col != ticker_col_name]
    df_list = [df.pivot(columns=ticker_col_name, values=ts) for ts in time_series]

    for i,df in enumerate(df_list):
        df.name = time_series[i]
        df.columns = [col + f"_{df.name}" for col in df.columns]

    df_dict = dict(zip(time_series, df_list))

    return df_dict


def detrend_time_series(frame: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """detrends a bag of time series"""

    for time_series, df in frame.items():
        frame[time_series] = frame[time_series].diff().dropna()

    return frame

@cached
def distance_correlation_matrix(df):

    tickers = df.columns.tolist()
    df_dcor = pd.DataFrame(index=tickers, columns=tickers)

    k=0
    for i in tickers:
        v_i = df.loc[:, i].values

        for j in tickers[k:]:
            v_j = df.loc[:, j].values
            dcor_val = dcor.distance_correlation(v_i, v_j)
            df_dcor.at[i, j] = dcor_val
            df_dcor.at[j, i] = dcor_val

        k+=1

    return df_dcor

@cached
def distance_correlation_network(df, corr_threshold=0.33):

    corr_matrix = df.values.astype('float')
    sim_matrix = 1 - corr_matrix

    G = nx.from_numpy_matrix(sim_matrix)
    ticker_names = df.index.values

    G = nx.relabel_nodes(G, lambda x: ticker_names[x])
    G.edges(data=True)

    H  = G.copy()

    for (u, v, wt) in G.edges.data('weight'):
        if wt >= 1 - corr_threshold:
            H.remove_edge(u, v)

        if u == v:
            H.remove_edge(u, v)

    return H

class HedgeFrame(object):
    def __init__(self, df: pd.DataFrame, ticker_col_name: str, detrend: bool=True) -> None:
        self.df = df
        self.ticker_col_name = ticker_col_name
        self.detrend = detrend

        if self.detrend:
            frame = detrend_time_series(self.preprocess())
        else:
            frame = self.preprocess()

        self.frame = frame

        def get_keys(frame: Dict[str, pd.DataFrame]) -> List[str]:
            keys = list(frame.keys())
            return keys

        keys = get_keys(self.frame)
        self.keys = keys

        self.first = frame[keys[0]]
        self.last = frame[keys[-1]]

    def preprocess(self) -> Dict[str, pd.DataFrame]:
        frame = ticker_time_frame(self.df, ticker_col_name=self.ticker_col_name)
        return frame

    def dcor(self) -> Dict[str, pd.DataFrame]:

        self.frame = {
            time_series: distance_correlation_matrix(df) for time_series, df in self.frame.items()
        }

        self.first = self.frame[self.keys[0]]
        self.last = self.frame[self.keys[-1]]

        return self

    def network(self, corr_threshold: int=0.33) -> Dict[str, pd.DataFrame]:

        self.frame = {
            time_series: distance_correlation_network(df_dcor, corr_threshold=corr_threshold)
            for time_series, df_dcor in self.frame.items()
        }

        self.first = self.frame[self.keys[0]]
        self.last = self.frame[self.keys[-1]]

        return self
