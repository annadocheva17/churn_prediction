from sklearn.feature_selection import SelectKBest, mutual_info_classif
from config import TOP_K_FEATURES

# Select only top k features to avoid overfitting
def select_top_features(X_train, y_train, k=TOP_K_FEATURES):
    
    print(f"\n Selecting top {k} features:")
    
    # Keep track of special columns
    # structural column that needs to be kept
    activity_col = ['has_activity']
    activity_mask = X_train.columns.isin(activity_col)
    
    X_for_selection = X_train.loc[:, ~activity_mask]
    
    if len(X_for_selection.columns) <= k:
        print(f"   Keeping all {len(X_train.columns)} features")
        return X_train.columns.tolist()
    
    # Use mutual information for feature selection
    # Mutual information: how much information about the target variable we gain by knowing the feature
    selector = SelectKBest(mutual_info_classif, k=k-len(activity_col))
    selector.fit(X_for_selection, y_train)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = X_for_selection.columns[selected_mask].tolist()
    
    # Add back activity column
    selected_features.extend(activity_col)
    
    print(f"Selected {len(selected_features)} features")
    
    return selected_features