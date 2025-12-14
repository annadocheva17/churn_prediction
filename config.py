from datetime import datetime, timedelta

# Paths
TRAIN_PATH = "data/train.parquet"
TEST_PATH = "data/test.parquet"
OUTPUT_PATH = "churn_predictions.csv"

# Time configuration
PREDICTION_START_DATE = datetime(2018, 11, 20)

OBSERVATION_WINDOW_DAYS = 21  # (obs_window_days)
PREDICTION_WINDOW_DAYS = 10
WINDOW_STEP_DAYS = 7  # (window_stride_days)

OBSERVATION_END_DATE = PREDICTION_START_DATE
OBSERVATION_START_DATE = OBSERVATION_END_DATE - timedelta(days=OBSERVATION_WINDOW_DAYS)

# Training / validation
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature selection
TOP_K_FEATURES = 30  # (top_k_features)
KEEP_SPECIAL_FEATURES = ["has_activity"]

# Optuna configuration
USE_OPTUNA = True
OPTUNA_N_TRIALS = 150
OPTUNA_TIMEOUT = 2400

OPTUNA_SAMPLER_SEED = 42
OPTUNA_N_FOLDS = 3

# XGBoost defaults (to be overridden by Optuna)
XGB_BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_MANUAL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 7,
    "learning_rate": 0.06,
    "min_child_weight": 2,
    "gamma": 0.05,
    "reg_alpha": 0.1,
    "reg_lambda": 1.2,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "scale_pos_weight": 20,
}

# Thresholding / inactive users
MIN_RECALL_TARGET = 0.55
DEFAULT_INACTIVE_CHURN_RATE = 0.30
