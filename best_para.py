import itertools
import time
import argparse
import warnings
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

# Import of previous fonctions 
try:
    from features import extract_features
    from labels import extract_window_labels
    from building_datasets import build_training_windows, build_training_dataset
    from preprocessing import preprocess_train
    from feature_selection import select_top_features
except ImportError:
    print("Error: ensure features.py, labels.py, preprocessing.py are present.")
    exit(1)  # exists if functions in files not found

warnings.filterwarnings("ignore")  # suppresses warinings for clean output

# Definition of the research

def run_meta_search(df, args):
    print()
    print(f"Starting grid search")
    
    # Generating combinations of parameters
    combos = []
    
    # Parse ranges helper
    def _range(x): return range(x[0], x[1], x[2]) if len(x) > 2 else range(x[0], x[1])

    # Generation of ranges for obs, nber features and stride
    print()
    print(f"Mode: Grid search (Obs: {args.obs}, TopK: {args.topk}, Stride: {args.stride})")
    obs_vals = _range(args.obs)
    topk_vals = _range(args.topk)
    stride_vals = _range(args.stride)
    for o, k, s in itertools.product(obs_vals, topk_vals, stride_vals):
        combos.append({"obs": int(o), "k": int(k), "stride": int(s)})

    # Optimizing: Group by (obs_window, stride) to avoid re-extracting data (quicker)
    combos.sort(key=lambda x: (x['obs'], x['stride']))
    grouped_combos = []
    for (obs, stride), group in itertools.groupby(combos, key=lambda x: (x['obs'], x['stride'])):
        # Collect all unique K values needed for this specific window config
        group_items = list(group)
        ks = sorted(list(set(g['k'] for g in group_items)))
        grouped_combos.append({"obs": obs, "stride": stride, "ks": ks})

    print(f"Nber of configurations: {len(combos)}")
    print(f"Nber of optimized groups: {len(grouped_combos)}")

    results = []
    best_score = 0.0
    start_global = time.time()

    # Execution of the loop
    for i, group in enumerate(grouped_combos, 1):
        obs_days = group['obs']
        stride = group['stride']
        ks_to_test = group['ks']
        
        # Extracting the data
        windows = build_training_windows(
            min_date=df["time"].min(),
            max_date=df["time"].max(),
            obs_window_days=obs_days,
            pred_horizon_days=10, 
            step_days=stride,
        )

        try:
            X_raw, y_full = build_training_dataset(
                df=df,
                windows=windows,
                feature_fn=extract_features,
                label_fn=extract_window_labels,
                cohort="pred_active",
            )
            if "userId" in X_raw.columns:
                groups = X_raw["userId"]
                X_raw = X_raw.drop(columns=["userId"])
            else:
                groups = X_raw.index.to_series()
            
            # Preprocessing
            groups.index = X_raw.index
            X_proc = preprocess_train(X_raw)

            y_full = y_full.loc[X_proc.index]
            groups = groups.loc[X_proc.index]
            
        except (RuntimeError, ValueError) as e:  # if nothing found
            print(f"Group [{i}/{len(grouped_combos)}] Skipped: {e}")
            continue

        # Cross-Validation & Feature Selection
        gkf = GroupKFold(n_splits=3)
        splits = list(gkf.split(X_proc, y_full, groups=groups))

        
        xgb_params = {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 6,
            "tree_method": "hist",
            "eval_metric": "auc",
            "n_jobs": -1,
            "random_state": 42
        }

        for k in ks_to_test:
            try:
                selected_features = select_top_features(X_proc, y_full, k=k)
                X_subset = X_proc[selected_features]
            except Exception as e:
                print(f"Feature selection failed for k={k}: {e}")
                continue

            fold_scores = []
            
            for tr_idx, va_idx in splits:
                X_tr, X_va = X_subset.iloc[tr_idx], X_subset.iloc[va_idx]
                y_tr, y_va = y_full.iloc[tr_idx], y_full.iloc[va_idx]

                model = XGBClassifier(**xgb_params)
                model.fit(X_tr, y_tr)
                
                try:
                    proba = model.predict_proba(X_va)[:, 1]
                    score = roc_auc_score(y_va, proba)
                    fold_scores.append(score)
                except Exception:
                    fold_scores.append(0.5)
            
            mean_auc = np.mean(fold_scores)
            
            # Tracking best score
            if mean_auc > best_score:
                best_score = mean_auc
                print(f"New best: {best_score:.4f}, with obs={obs_days}, k={k} and stride={stride}")
            
            results.append({
                "obs_window_days": obs_days,
                "top_k_features": k,
                "window_stride_days": stride,
                "mean_auc": mean_auc
            })

        # Updating the progress
        elapsed = time.time() - start_global
        eta = (elapsed / i) * (len(grouped_combos) - i) / 60
        print(f"Group [{i}/{len(grouped_combos)}] done. Time until end: {eta:.1f} min")

    return pd.DataFrame(results)

# Running the research

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.parquet")
    parser.add_argument("--results", default="para_search_results.csv")
    
    # Grid configuration [min, max, step]
    parser.add_argument("--obs", nargs='+', type=int, default=[14, 45, 7])
    parser.add_argument("--topk", nargs='+', type=int, default=[10, 40, 10])
    parser.add_argument("--stride", nargs='+', type=int, default=[7, 14, 7])

    args = parser.parse_args()

    # Loading data
    print()
    print(f"Loading {args.data}...")
    try:
        df = pd.read_parquet(args.data)
        df["time"] = pd.to_datetime(df["time"])
        if "registration" in df.columns:
            df["registration"] = pd.to_datetime(df["registration"])
    except Exception as e:
        print(f"Error loading data: {e}")  # if files not found
        exit(1)

    # Running function
    res = run_meta_search(df, args)
    
    if not res.empty:
        res = res.sort_values("mean_auc", ascending=False)
        print()
        print("="*40)
        print(f"Done, best AUC: {res['mean_auc'].max():.4f}")
        print("="*40)
        print("Top 5 Results:")
        print(res.head(5).to_string(index=False))
        res.to_csv(args.results, index=False)
        print()
        print(f"Saved to {args.results}")
    else:
        print("No results found, check the data or ranges")