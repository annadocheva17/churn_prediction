import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

from config import (
    OUTPUT_PATH,
    OBSERVATION_START_DATE,
    OBSERVATION_END_DATE,
    OBSERVATION_WINDOW_DAYS,
    PREDICTION_WINDOW_DAYS,
    WINDOW_STEP_DAYS,
    TEST_SIZE,
    RANDOM_STATE,
    TOP_K_FEATURES,
    DEFAULT_INACTIVE_CHURN_RATE,
)

from data_loading import load_train, load_test
from building_datasets import build_training_windows, build_training_dataset, build_test_features
from features import extract_features
from labels import extract_window_labels
from preprocessing import preprocess_train, preprocess_test
from feature_selection import select_top_features
from model_xgb import optimize_hyperparameters
from evaluation import choose_threshold_min_recall


def main():
    print()
    print("IMPROVED CHURN PREDICTION - FEATURE SELECTION + INACTIVE STRATEGY")
    print()

    # Loading data
    print("Loading training data...")
    train_df = load_train()
    print(f"{len(train_df):,} events, {train_df['userId'].nunique():,} users")

    # Build rolling windows
    print()
    print("Creating training windows...")
    windows = build_training_windows(
        min_date=train_df["time"].min(),
        max_date=train_df["time"].max(),
        obs_window_days=OBSERVATION_WINDOW_DAYS,
        pred_horizon_days=PREDICTION_WINDOW_DAYS,
        step_days=WINDOW_STEP_DAYS,
    )
    print(f"Created {len(windows)} training windows")

    # Build training dataset
    print()
    print("Building training dataset...")
    X_train_raw, y_train = build_training_dataset(
        df=train_df,
        windows=windows,
        feature_fn=extract_features,
        label_fn=extract_window_labels,
        cohort="pred_active",
    )
    
    # CHANGE: Extract groups (userIds) before preprocessing removes them
    if 'userId' in X_train_raw.columns:
        train_groups = X_train_raw['userId']
    else:
        train_groups = X_train_raw.index.to_series()
        
    print(f"Samples: {len(X_train_raw):,}, churn rate: {y_train.mean():.2%}, raw feats: {X_train_raw.shape[1]}")

    # Inactive analysis
    print()
    inactive_mask = (X_train_raw.get("has_activity", 0) == 0)
    if inactive_mask.sum() > 0:
        inactive_churn_rate = float(y_train[inactive_mask].mean())
    else:
        inactive_churn_rate = float(DEFAULT_INACTIVE_CHURN_RATE)
    print(f"Inactive churn rate (train): {inactive_churn_rate:.2%}")

    # Preprocess & feature selection
    print()
    print("Preprocessing...")
    X_train_proc = preprocess_train(X_train_raw)
    selected_features = select_top_features(X_train_proc, y_train, k=TOP_K_FEATURES)
    X_train_sel = X_train_proc[selected_features]
    print(f"Final feature count: {X_train_sel.shape[1]}")

    # Hyperparams, validation & threshold
    # CHANGE: Pass groups to optimizer
    best_params = optimize_hyperparameters(X_train_sel, y_train, groups=train_groups)
    
    print()
    print("Training model with selected features (Grouped Split)...")
    
    # CHANGE: Use GroupShuffleSplit instead of train_test_split
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss.split(X_train_sel, y_train, groups=train_groups))
    
    X_tr = X_train_sel.iloc[train_idx]
    y_tr = y_train.iloc[train_idx]
    X_val = X_train_sel.iloc[val_idx]
    y_val = y_train.iloc[val_idx]

    tmp_model = XGBClassifier(**best_params)
    tmp_model.fit(X_tr, y_tr)

    val_proba = tmp_model.predict_proba(X_val)[:, 1]
    print(f"Validation AUC: {roc_auc_score(y_val, val_proba):.4f}")

    best_threshold, info = choose_threshold_min_recall(y_val, val_proba)
    print(f"Threshold: {best_threshold:.3f} ({info['chosen_by']})")
    print(f"Precision: {info['precision_at_threshold']:.4f} | Recall: {info['recall_at_threshold']:.4f}")

    # Test features
    print()
    print("Loading test data...")
    test_df = load_test()
    print(f"{len(test_df):,} events, {test_df['userId'].nunique():,} users")
    print()
    print("Extracting test features...")
    test_features = build_test_features(
        df=test_df,
        window_start=OBSERVATION_START_DATE,
        window_end=OBSERVATION_END_DATE,
        feature_fn=extract_features,
    )

    test_user_ids = test_features["userId"].copy()
    is_inactive_test = (test_features.get("has_activity", 0) == 0)
    print(f"Total: {len(test_user_ids)}, Inactive: {int(is_inactive_test.sum())} ({is_inactive_test.mean()*100:.1f}%)")

    X_test_raw = test_features.drop(columns=["userId"])
    X_test_sel = preprocess_test(X_test_raw, selected_features)

    # Final train & predict
    print("Training on full data and predicting...")
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train_sel, y_train)

    test_proba = final_model.predict_proba(X_test_sel)[:, 1]
    predictions = (test_proba >= best_threshold).astype(int)

    # Stategy for inactive users
    rng = np.random.RandomState(RANDOM_STATE)
    inactive_idx = np.where(is_inactive_test.values)[0]
    if len(inactive_idx) > 0:
        predictions[inactive_idx] = (rng.rand(len(inactive_idx)) < inactive_churn_rate).astype(int)
    print()
    print(f"Prediction Summary:")
    print(f"Total predicted churns: {int(predictions.sum())} ({predictions.mean():.2%})")

    # Saving submission
    submission = pd.DataFrame({"id": test_user_ids, "target": predictions})
    submission.to_csv(OUTPUT_PATH, index=False)
    print()
    print(f"Saved to '{OUTPUT_PATH}' ! ")


if __name__ == "__main__":
    main()