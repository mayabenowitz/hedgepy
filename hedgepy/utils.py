from typing import Dict
import json
import pandas as pd

def write(df_ts: Dict[pd.Timestamp, pd.DataFrame], file_name: str) -> None:
    keys = list(df_ts.keys())
    values = list(df_ts.values())

    keys = map(lambda timestamp: str(timestamp), keys)
    values = map(lambda df: df.to_dict('records'), values)

    df_ts = dict(zip(keys, values))
    with open(f'../experiments/data/interim/{file_name}', 'w') as f:
        json.dump(df_ts, f)

def read(json_file: str) -> Dict[pd.Timestamp, pd.DataFrame]:
    with open(f'../experiments/data/interim/{json_file}.json') as f:
        data = json.load(f)

    keys = list(data.keys())
    values = list(data.values())

    keys = map(lambda k: pd.Timestamp(k), keys)
    values = map(lambda v: pd.DataFrame.from_dict(v), values)

    df_ts = dict(zip(keys, values))
    return df_ts
