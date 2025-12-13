import pandas as pd
import numpy as np
from sympy import re

def clean_column_names(df):
    """Clean column names to be XGBoost compatible."""
    df = df.copy()
    df.columns = df.columns.str.replace('[', '_', regex=False)
    df.columns = df.columns.str.replace(']', '_', regex=False)
    df.columns = df.columns.str.replace('<', 'lt', regex=False)
    df.columns = df.columns.str.replace('>', 'gt', regex=False)
    df.columns = df.columns.str.replace(' ', '_', regex=False)
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]
    return df

def extract_features(df, window_start, window_end, include_activity_flag=True):
    window_df = df[(df['time'] >= window_start) & (df['time'] < window_end)].copy()
    
    if len(window_df) == 0:
        return pd.DataFrame()
    
    window_df = window_df.sort_values('time')
    
    # Page interaction dummies
    page_dummies = pd.get_dummies(window_df['page'], prefix='page')
    window_df = pd.concat([window_df, page_dummies], axis=1)
    
    # Basic aggregations
    features = window_df.groupby('userId').agg({
        'sessionId': 'nunique',
        'itemInSession': 'sum',
        'song': 'nunique',
        'artist': 'nunique',
        'length': ['sum', 'mean'],
        'gender': 'first',
        'level': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'registration': 'first',
    }).reset_index()
    
    features.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                              for col in features.columns]
    
    # Page interaction features
    key_pages = [
        'page_Thumbs Down',
        'page_Thumbs Up', 
        'page_Downgrade',
        'page_Roll Advert',
        'page_Logout',
        'page_Submit Upgrade',
        'page_Add Friend',
        'page_Error',
        'page_Submit Downgrade',
        'page_Add to Playlist',
        'page_Settings',
        'page_Home',
        'page_NextSong'
    ]
    
    page_cols = [col for col in window_df.columns if col in key_pages]
    if page_cols:
        page_features = window_df.groupby('userId')[page_cols].sum().reset_index()
        features = features.merge(page_features, on='userId', how='left')
        features[page_cols] = features[page_cols].fillna(0)

    # Engagement trends over thirds of the window
    window_duration = window_end - window_start
    third_1 = window_start + window_duration / 3
    third_2 = window_start + 2 * window_duration / 3
    
    first_third = window_df[window_df['time'] < third_1]
    second_third = window_df[(window_df['time'] >= third_1) & (window_df['time'] < third_2)]
    third_third = window_df[window_df['time'] >= third_2]
    
    first_sessions = first_third.groupby('userId')['sessionId'].nunique()
    second_sessions = second_third.groupby('userId')['sessionId'].nunique()
    third_sessions = third_third.groupby('userId')['sessionId'].nunique()
    
    engagement_trends = pd.DataFrame({
        'userId': features['userId'],
        'sessions_third_1': features['userId'].map(first_sessions).fillna(0),
        'sessions_third_2': features['userId'].map(second_sessions).fillna(0),
        'sessions_third_3': features['userId'].map(third_sessions).fillna(0),
    })
    
    # Decline indicators
    engagement_trends['early_to_mid_decline'] = (
        engagement_trends['sessions_third_1'] - engagement_trends['sessions_third_2']
    )
    engagement_trends['mid_to_late_decline'] = (
        engagement_trends['sessions_third_2'] - engagement_trends['sessions_third_3']
    )
    engagement_trends['total_decline'] = (
        engagement_trends['sessions_third_1'] - engagement_trends['sessions_third_3']
    )
    engagement_trends['consistent_decline'] = (
        (engagement_trends['early_to_mid_decline'] > 0) & 
        (engagement_trends['mid_to_late_decline'] > 0)
    ).astype(int)
    
    features = features.merge(engagement_trends, on='userId', how='left')
    
    # Days active
    days_active = window_df.groupby('userId')['time'].apply(
        lambda x: x.dt.date.nunique()
    ).reset_index()
    days_active.columns = ['userId', 'days_active']
    features = features.merge(days_active, on='userId', how='left')
    
    # Last activity time
    last_activity = window_df.groupby('userId')['time'].max().reset_index()
    last_activity.columns = ['userId', 'last_activity_time']
    features = features.merge(last_activity, on='userId', how='left')
    features['days_since_last_activity'] = (
        (window_end - features['last_activity_time']).dt.total_seconds() / 86400
    )
    
    # Days since registration
    features['days_since_registration'] = (
        (window_start - features['registration_first']).dt.total_seconds() / 86400
    )
    
    # Ratios (per session/day)
    features['songs_per_session'] = (
        features['song_nunique'] / (features['sessionId_nunique'] + 1)
    )
    features['items_per_session'] = (
        features['itemInSession_sum'] / (features['sessionId_nunique'] + 1)
    )
    features['songs_per_day'] = (
        features['song_nunique'] / (features['days_active'] + 1)
    )
    features['sessions_per_day'] = (
        features['sessionId_nunique'] / (features['days_active'] + 1)
    )
    
    # Engagement velocity
    features['session_velocity'] = (
        features['sessions_third_3'] / (features['sessions_third_1'] + 1)
    )
    
    # thumbs down ratio
    thumbs_up = features.get('page_Thumbs Up', 0)
    thumbs_down = features.get('page_Thumbs Down', 0)
    features['thumbs_down_ratio'] = thumbs_down / (thumbs_up + thumbs_down + 1)
    
    # Activity flag
    if include_activity_flag:
        features['has_activity'] = 1
    
    # Drop intermediate columns
    drop_cols = ['registration_first', 'last_activity_time']
    features = features.drop([c for c in drop_cols if c in features.columns], axis=1)
    
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    
    return features