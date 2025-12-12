import pandas as pd
import numpy as np

def extract_window_labels(df, window_start, window_end):
    window_df = df[(df['time'] >= window_start) & (df['time'] < window_end)].copy()
    
    if len(window_df) == 0:
        return pd.DataFrame()
    
    churned_users = window_df[
        window_df['page'] == 'Cancellation Confirmation'
    ]['userId'].unique()
    
    all_users = window_df['userId'].unique()
    
    return pd.DataFrame({
        'userId': all_users,
        'churned': np.isin(all_users, churned_users).astype(int)
    })
