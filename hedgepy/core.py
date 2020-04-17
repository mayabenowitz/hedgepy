from typing import Dict
import pandas as pd


def ticker_time_frame(
    df: pd.DataFrame, ticker_col_name: str
) -> Dict[str, pd.DataFrame]:
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


def detrend(frame: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """detrends a bag of time series"""

    for time_series, df in frame.items():
        frame[time_series] = frame[time_series].diff().dropna()

    return frame


class HedgeFrame(object):
    def __init__(self, df: pd.DataFrame, ticker_col_name: str, detrend: bool=True) -> None:
        self.df = df
        self.ticker_col_name = ticker_col_name
        self.detrend = detrend
        self.preprocess()

    def preprocess(self, detrend: bool=detrend) -> Dict[str, pd.DataFrame]:
        frame = ticker_time_frame(self.df, ticker_col_name=self.ticker_col_name)

        if detrend:
            frame = detrend(frame)

        keys = list(frame.keys())

        self.frame = frame
        self.first = frame[keys[0]]
        self.last = frame[keys[-1]]
        return frame
