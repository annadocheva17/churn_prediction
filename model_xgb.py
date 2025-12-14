import numpy as np
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, recall_score

from config import (
    USE_OPTUNA,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    OPTUNA_SAMPLER_SEED,
    OPTUNA_N_FOLDS,
    RANDOM_STATE,
    XGB_BASE_PARAMS,
    XGB_MANUAL_PARAMS,
)


def objective(trial, X_train, y_train, groups, n_folds: int = OPTUNA_N_FOLDS):
    params = {
        **XGB_BASE_PARAMS,
        "n_estimators": trial.suggest_int("n_estimators", 150, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
        "gamma": trial.suggest_float("gamma", 0.0, 0.3),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        "subsample": trial.suggest_float("subsample", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.95),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 10, 30),
        "random_state": RANDOM_STATE,  
    }

    # CHANGE: Use GroupKFold instead of StratifiedKFold to ensure there is no data leakage
    gkf = GroupKFold(n_splits=n_folds)

    aucs = []
    recalls_at_05 = []

    
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups=groups)):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, verbose=False)

        proba = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, proba)
        aucs.append(auc)

        # track recall at 0.5 threshold
        pred_05 = (proba >= 0.5).astype(int)
        recalls_at_05.append(recall_score(y_va, pred_05))

        trial.report(auc, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("mean_recall_at_0.5", float(np.mean(recalls_at_05)))
    return float(np.mean(aucs))


def optimize_hyperparameters(X_train, y_train, groups): 
    if (not USE_OPTUNA) or (optuna is None):
        params = {**XGB_BASE_PARAMS, **XGB_MANUAL_PARAMS, "random_state": RANDOM_STATE}
        return params

    sampler = TPESampler(seed=OPTUNA_SAMPLER_SEED)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, groups),
        n_trials=OPTUNA_N_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        n_jobs=1,
        show_progress_bar=True,
    )

    best = study.best_params
    best_params = {**XGB_BASE_PARAMS, **best, "random_state": RANDOM_STATE}

    return best_params


def train_final_xgb(X_train, y_train, best_params: dict):
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)
    return model