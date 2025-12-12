import pandas as pd
from config import TRAIN_PATH, TEST_PATH


def load_train(path: str = TRAIN_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df["registration"] = pd.to_datetime(df["registration"])
    return df


def load_test(path: str = TEST_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df["registration"] = pd.to_datetime(df["registration"])
    return df