import pandas as pd


def load_train(path: str = "data/train.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df["registration"] = pd.to_datetime(df["registration"])
    return df


def load_test(path: str = "data/test.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df["registration"] = pd.to_datetime(df["registration"])
    return df