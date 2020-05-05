from typing import Dict
import json
import pandas as pd

def write_series(df_ts: Dict[pd.Timestamp, pd.DataFrame], file_name: str) -> None:
    keys = list(df_ts.keys())
    values = list(df_ts.values())

    keys = map(lambda timestamp: str(timestamp), keys)
    values = map(lambda df: df.to_dict('records'), values)

    df_ts = dict(zip(keys, values))
    with open(f'../experiments/data/interim/{file_name}', 'w') as f:
        json.dump(df_ts, f)

def read_series(json_file: str) -> Dict[pd.Timestamp, pd.DataFrame]:
    with open(f'../experiments/data/interim/{json_file}.json') as f:
        data = json.load(f)

    keys = list(data.keys())
    values = list(data.values())

    keys = map(lambda k: pd.Timestamp(k), keys)
    values = map(lambda v: pd.DataFrame.from_dict(v), values)

    df_ts = dict(zip(keys, values))
    return df_ts

def read_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(f"../experiments/data/interim/{csv_file}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.dropna(inplace=True)

    return df
