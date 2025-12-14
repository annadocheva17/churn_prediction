import pandas as pd
import h2o
from h2o.automl import H2OAutoML
import warnings

from config import (
    OBSERVATION_WINDOW_DAYS,
    PREDICTION_WINDOW_DAYS,
    WINDOW_STEP_DAYS,
    OBSERVATION_START_DATE,
    OBSERVATION_END_DATE,
)
from data_loading import load_train, load_test
from building_datasets import (
    build_training_windows,
    build_training_dataset,
    build_test_features,
)
from features import extract_features
from labels import extract_window_labels
from preprocessing import clean_column_names

warnings.filterwarnings('ignore')

def main():
    print()
    print("H2O AUTOML CHURN PREDICTION")
    print()

    # Loading data
    print("Loading data...")
    train_df = load_train()
    test_df = load_test()
    print(f"Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")

    # Building training data
    print()
    print("Building training dataset...")
    windows = build_training_windows(
        min_date=train_df["time"].min(),
        max_date=train_df["time"].max(),
        obs_window_days=OBSERVATION_WINDOW_DAYS,
        pred_horizon_days=PREDICTION_WINDOW_DAYS,
        step_days=WINDOW_STEP_DAYS,
    )

    X_train, y_train = build_training_dataset(
        df=train_df,
        windows=windows,
        feature_fn=extract_features,
        label_fn=extract_window_labels,
        cohort="pred_active",
    )
    
    # H2O needs clean column names without special char
    X_train = clean_column_names(X_train)
    print(f"Training samples: {len(X_train):,}")

    # Building test features
    print()
    print("Extracting test features...")
    test_features = build_test_features(
        df=test_df,
        window_start=OBSERVATION_START_DATE,
        window_end=OBSERVATION_END_DATE,
        feature_fn=extract_features,
    )
    
    test_user_ids = test_features["userId"]
    X_test = test_features.drop(columns=["userId"])
    X_test = clean_column_names(X_test)

    # Aligning columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # H2O AutoML
    print()
    print("Starting H2O AutoML...")
    h2o.init(max_mem_size='8G')

    # Prepare H2O frames (needs X and y in the same frame)
    train_h2o = h2o.H2OFrame(pd.concat([X_train, y_train.rename("target")], axis=1))
    test_h2o = h2o.H2OFrame(X_test)

    # Ensure target treated as categorical
    train_h2o["target"] = train_h2o["target"].asfactor()

    aml = H2OAutoML(
        max_runtime_secs=600,
        max_models=20,
        seed=42,
        stopping_metric='AUC',
        sort_metric='AUC',
        balance_classes=True,
        verbosity='info'
    )
    print()
    print("Training...")
    aml.train(y="target", training_frame=train_h2o)

    # Results
    print()
    print("Leaderboard:")
    print(aml.leaderboard.head().as_data_frame().to_string(index=False))
    
    h2o.cluster().shutdown(prompt=False)

if __name__ == "__main__":
    main()
