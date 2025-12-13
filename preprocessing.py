import numpy as np
import pandas as pd
import re

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.replace("[", "_", regex=False)
    df.columns = df.columns.str.replace("]", "_", regex=False)
    df.columns = df.columns.str.replace("<", "lt", regex=False)
    df.columns = df.columns.str.replace(">", "gt", regex=False)
    df.columns = df.columns.str.replace(" ", "_", regex=False)
    df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", c) for c in df.columns]
    return df


def preprocess_train(X_train):
    X = X_train.copy()

    # One-hot encoding categorical variables
    cat_cols = X.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Replace invalid values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Clean column names
    X = clean_column_names(X)

    return X


def preprocess_test(X_test, selected_features):
    X = X_test.copy()

    # One-hot encode
    X = pd.get_dummies(X)

    # Clean column names
    X = clean_column_names(X)

    # Align with training features
    X = X.reindex(columns=selected_features, fill_value=0)

    # Replace invalid values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X
