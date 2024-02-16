import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def create_forward_features(gw_data_frames, up_to_gw):
    """
    Create forward features DataFrame up to a specified gameweek.

    Parameters:
    - gw_data_frames: Dictionary of DataFrames for each gameweek.
    - up_to_gw: Integer, the last gameweek to include in the feature calculation.

    Returns:
    - DataFrame with forward features calculated up to the specified gameweek.
    """
    # Combine DataFrames up to the specified gameweek
    combined_df = pd.concat([df for gw, df in gw_data_frames.items() if gw <= up_to_gw], ignore_index=True)

    # Filter for forwards
    df_forward = combined_df.loc[combined_df['position'] == 'FWD']

    # Group by player name
    forward_grouped = df_forward.groupby('name')

    # Initialize the features DataFrame
    forward_features = pd.DataFrame(index=forward_grouped.indices.keys())

    # Calculate features relevant to forwards
    forward_features['minutes'] = forward_grouped['minutes'].mean()
    forward_features['goals_scored'] = forward_grouped['goals_scored'].sum()  # Key feature, as forwards earn points mostly through goals
    forward_features['assists'] = forward_grouped['assists'].sum()  # Assists are also a significant point source
    forward_features['bonus'] = forward_grouped['bonus'].mean()  # Forwards often earn bonus points through goal contributions
    forward_features['yellow_cards'] = forward_grouped['yellow_cards'].sum()  # Points deduction for yellow cards
    forward_features['red_cards'] = forward_grouped['red_cards'].sum()  # Points deduction for red cards
    forward_features['own_goals'] = forward_grouped['own_goals'].sum()  # Points deduction for own goals

    # Calculate recent form (total points in the last 5 games)
    recent_form = []
    for name, group in forward_grouped:
        group = group.sort_values(by='gw', ascending=False)
        recent_form.append(group['total_points'].head(5).sum())
    forward_features['recent_form'] = recent_form

    return forward_features

def train_and_evaluate_rf(gw_data_frames, target_gw):
    """
    Train a Random Forest model with data up to a specified gameweek and evaluate it using the target gameweek.
    Automatically excludes defenders not present in the feature set.

    Parameters:
    - gw_data_frames: Dictionary of DataFrames for each gameweek.
    - target_gw: Integer, the gameweek to predict and evaluate the model.

    Returns:
    - mse: Mean Squared Error of the model predictions for the target gameweek.
    """
    
    forward_features = create_forward_features(gw_data_frames, target_gw - 1)
    
    # Load target gameweek data
    gw_target_data = gw_data_frames[target_gw]
    gw_target_defender = gw_target_data.loc[gw_target_data['position'] == 'FWD']
    
    # Prepare target variable
    y = gw_target_defender[['name', 'total_points']].set_index('name').sort_index()

    # Ensure we only consider defenders present in both the features and target sets
    common_indices = forward_features.index.intersection(y.index)
    x_filtered = forward_features.loc[common_indices]
    y_filtered = y.loc[common_indices]

    # Check if sufficient data is available for split
    if len(x_filtered) < 5 or len(y_filtered) < 5:
        print(f"Insufficient data for gameweek {target_gw}. Skipping.")
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(x_filtered, y_filtered, test_size=0.2, shuffle=False)
    except ValueError as e:
        print(f"Error splitting data for gameweek {target_gw}: {e}")
        return None

    # Train the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train.values.ravel())  # .values.ravel() to convert y_train to 1D array if needed

    # Predict and evaluate
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse