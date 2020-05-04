from typing import Dict, List, Union
import pandas as pd
import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import expm
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

    df_dict = dict(zip(time_series, df_list))

    return df_dict


def detrend_time_series(frame: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """detrends a bag of time series"""

    for time_series, df in frame.items():
        frame[time_series] = frame[time_series].diff().dropna()

    return frame


def coalesce_time_series(frame: Dict[str, pd.DataFrame], rolling: bool=False) -> pd.DataFrame:

    if not rolling:
        frame = {
                   time_series: df.applymap(lambda x: [x]) for time_series, df in frame.items()
            }

        frame_lst = list(frame.values())
        coalesced_frame = frame_lst[0]

        for frame in frame_lst[1:]:
            dfc = coalesced_frame + frame

        dfc = dfc.applymap(lambda x: np.array(x)).dropna()
        return dfc

    if rolling:
        frame = {
            time_series:

                [
                    frame[time_series][i].applymap(lambda x: [x])
                    for i,df in enumerate(rolling_df_list)
                ]

            for time_series, rolling_df_list in frame.items()
        }

        dff = pd.DataFrame.from_dict(frame)
        col1 = dff.columns[0]
        for col in dff.columns[1:]:
            dff[col1] += dff[col]

        cdf_list = dff[col1].tolist()
        cdf_list = [df.applymap(lambda x: np.array(x)).dropna() for df in cdf_list]
        timestamps = [cdf_list[i].index[-1] for i in range(len(cdf_list))]

        frame = dict(zip(timestamps, cdf_list))
        return frame


@cached
def distance_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:

    tickers = df.columns.tolist()
    df_dcor = pd.DataFrame(index=tickers, columns=tickers)

    k=0
    for i in tickers:
        v_i = df.loc[:, i].values
        v_i = np.array([i for i in v_i])

        for j in tickers[k:]:
            v_j = df.loc[:, j].values
            v_j = np.array([j for j in v_j])
            dcor_val = dcor.distance_correlation(v_i, v_j)
            df_dcor.at[i, j] = dcor_val
            df_dcor.at[j, i] = dcor_val

        k+=1

    # dcor_matrix = matrix_power(df_dcor.to_numpy(), 2)
    # dcor_matrix = expm(df_dcor.to_numpy())
    # df_expdcor = pd.DataFrame(dcor_matrix)
    # df_expdcor.columns = df_dcor.columns
    # df_expdcor.index = df_dcor.index

    # return df_expdcor
    return df_dcor

@cached
def build_correlation_network(df: pd.DataFrame, corr_threshold=None, soft_threshold=True) -> nx.Graph:

    if soft_threshold:
        df_exp = expm(df.to_numpy())
        df_exp = pd.DataFrame(df_exp)
        df_exp.columns = df.columns
        df_exp.index = df.index
        corr_matrix = df_exp.values.astype('float')
    else:
        corr_matrix = df.values.astype('float')
    # sim_matrix = 1 - corr_matrix

    G = nx.from_numpy_matrix(corr_matrix)
    ticker_names = df.index.values

    G = nx.relabel_nodes(G, lambda x: ticker_names[x])
    G.edges(data=True)

    H  = G.copy()

    for (u,v,wt) in G.edges.data('weight'):
        if u == v:
            H.remove_edge(u, v)

    if corr_threshold is not None:
        for (u, v, wt) in G.edges.data('weight'):
            if wt <= corr_threshold:
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

        def get_keys(frame: Dict[str, pd.DataFrame]) -> List[str]:
            keys = list(frame.keys())
            return keys

        keys = get_keys(self.frame)
        self.keys = keys
        self.first = frame[keys[0]]
        self.last = frame[keys[-1]]

    def preprocess(self) -> Dict[str, pd.DataFrame]:
        frame = ticker_time_frame(self.df, ticker_col_name=self.ticker_col_name)
        self.frame = frame
        return frame

    def dcor(self, rolling_window=None, coalesce=True) -> Dict[str, pd.DataFrame]:

        if rolling_window is None:
            if coalesce:
                frame = coalesce_time_series(frame, rolling=False)
                frame = distance_correlation_matrix(frame)
                self.frame = frame
            if not coalesce:
                frame = {
                    time_series: distance_correlation_matrix(df)
                    for time_series, df in self.frame.items()
                }
                self.frame = frame
            return frame

        else:
            frame = {
                time_series: [df.iloc[i:i+rolling_window] for i in range(len(df)-rolling_window)]
                for time_series, df in self.frame.items()
            }
            if coalesce:
                frame = coalesce_time_series(frame, rolling=True)
                frame = {
                    timestamp: distance_correlation_matrix(df)
                    for timestamp, df in frame.items()
                }
                self.frame = frame
            if not coalesce:
                frame = {
                    time_series:

                        [
                            distance_correlation_matrix(frame[time_series][i])
                            for i,df in enumerate(rolling_df_list)
                        ]

                    for time_series, rolling_df_list in frame.items()
                }
                self.frame = frame

            return frame

@cached
def build_series(df, ticker_col_name, rolling_window, coalesce=True, detrend=True):
    hf = HedgeFrame(df, ticker_col_name=ticker_col_name, detrend=detrend)
    frame = hf.dcor(rolling_window=rolling_window, coalesce=coalesce)

    return frame

def build_network_time_series(frame: Dict[str, pd.DataFrame], corr_threshold=None, soft_threshold: bool=True) -> Dict[pd.Timestamp, nx.Graph]:
    frame = {
        time_series: build_correlation_network(df_dcor, corr_threshold=corr_threshold, soft_threshold=soft_threshold)
        for time_series, df_dcor in frame.items()
    }
    return frame



    # def network(self, corr_threshold=None) -> Dict[str, pd.DataFrame]:
    #
    #     frame = {
    #         time_series: build_correlation_network(df_dcor, corr_threshold=corr_threshold)
    #         for time_series, df_dcor in self.frame.items()
    #     }
    #     self.frame = frame
    #
    #     return self
