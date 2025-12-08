import pandas as pd
from datetime import timedelta


obs_window_days = 14
pred_horizon_days = 10


def build_training_windows(min_date, max_date, step_days: int = 7):
    windows = []
    current_start = min_date
    while True:
        obs_start = current_start
        obs_end = obs_start + timedelta(days=obs_window_days)
        pred_start = obs_end
        pred_end = pred_start + timedelta(days=pred_horizon_days)
        if pred_end > max_date:
            break
        windows.append(
            {
                "obs_start": obs_start,
                "obs_end": obs_end,
                "pred_start": pred_start,
                "pred_end": pred_end,
            }
        )
        current_start += timedelta(days=step_days)
    return windows


def build_training_dataset(
    df: pd.DataFrame,
    windows,
    feature_fn,
    label_fn,
    w2v_model=None,
    embedding_dim: int = 0,
):
    X_list = []
    y_list = []

    for w in windows:
        pred_window_df = df[
            (df["time"] >= w["pred_start"]) & (df["time"] < w["pred_end"])
        ]
        all_pred_users = pred_window_df["userId"].unique()
        if len(all_pred_users) == 0:
            continue

        features = feature_fn(
            df,
            w["obs_start"],
            w["obs_end"],
            w2v_model=w2v_model,
            embedding_dim=embedding_dim,
        )
        labels = label_fn(df, w["pred_start"], w["pred_end"])
        if len(labels) == 0:
            continue

        all_users_df = pd.DataFrame({"userId": all_pred_users})
        dataset = all_users_df.merge(features, on="userId", how="left")
        dataset = dataset.merge(labels, on="userId", how="inner")
        if len(dataset) == 0:
            continue

        dataset = dataset.fillna(0)
        X = dataset.drop(["userId", "churned"], axis=1)
        y = dataset["churned"]

        X_list.append(X)
        y_list.append(y)

    if not X_list:
        raise RuntimeError("No training data constructed from windows.")

    X_train = pd.concat(X_list, ignore_index=True)
    y_train = pd.concat(y_list, ignore_index=True)
    return X_train, y_train


def build_test_features(
    df: pd.DataFrame,
    window_start,
    window_end,
    feature_fn,
    w2v_model=None,
    embedding_dim: int = 0,
):
    all_users = df["userId"].unique()
    features = feature_fn(
        df,
        window_start,
        window_end,
        w2v_model=w2v_model,
        embedding_dim=embedding_dim,
    )
    all_users_df = pd.DataFrame({"userId": all_users})
    complete = all_users_df.merge(features, on="userId", how="left")
    complete = complete.fillna(0)
    return complete
