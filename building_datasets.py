import pandas as pd
from datetime import timedelta


def build_training_windows(
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    obs_window_days: int,
    pred_horizon_days: int,
    step_days: int,
):
    windows = []
    current_start = pd.to_datetime(min_date)
    max_date = pd.to_datetime(max_date)

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
    cohort: str = "obs_active",
):
    X_list = []
    y_list = []

    time_col = df["time"]

    for w in windows:
        obs_mask = (time_col >= w["obs_start"]) & (time_col < w["obs_end"])
        pred_mask = (time_col >= w["pred_start"]) & (time_col < w["pred_end"])

        if cohort == "obs_active":
            users = df.loc[obs_mask, "userId"].unique()
        elif cohort == "pred_active":
            users = df.loc[pred_mask, "userId"].unique()
        else:
            raise ValueError("cohort must be 'obs_active' or 'pred_active'")

        if len(users) == 0:
            continue

        features = feature_fn(
            df,
            w["obs_start"],
            w["obs_end"],
        )

        labels = label_fn(df, w["pred_start"], w["pred_end"])
        if len(labels) == 0:
            continue

        cohort_df = pd.DataFrame({"userId": users})

        dataset = cohort_df.merge(features, on="userId", how="left")
        dataset = dataset.merge(labels, on="userId", how="left")

        if "churned" not in dataset.columns:
            raise ValueError("label_fn must return columns ['userId','churned']")

        dataset["churned"] = dataset["churned"].fillna(0).astype(int)
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
):
    all_users = df["userId"].unique()

    features = feature_fn(
        df,
        window_start,
        window_end,
    )

    all_users_df = pd.DataFrame({"userId": all_users})
    complete = all_users_df.merge(features, on="userId", how="left")
    complete = complete.fillna(0)
    return complete
