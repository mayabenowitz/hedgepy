from typing import Dict
import pandas as pd


def bag_of_time_series(
    df: pd.DataFrame, ticker_col_name: str
) -> Dict[str, pd.DataFrame]:
    """Returns a dictionary of pivoted dataframes"""

    try:
        assert isinstance(df.index, pd.DatetimeIndex)
    except AssertionError:
        print("Please pass a DataFrame with a DatetimeIndex")

    time_series = df.columns.tolist()

    df_list = [
        df.pivot(columns=ticker_col_name, values=ts)
        for ts in time_series
        if ts != ticker_col_name
    ]
    df_dict = dict(zip(time_series, df_list))

    return df_dict


def detrend(bag: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """detrends a bag of time series"""

    for time_series, df in bag.items():
        bag[time_series] = bag[time_series].diff()

    return bag


class HedgeFrame(object):
    def __init__(self, df: pandas.DataFrame) -> None:
        self.df = df
