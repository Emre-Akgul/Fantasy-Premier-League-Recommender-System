import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def create_defender_features(gw_data_frames, up_to_gw):
    """
    Create defender features DataFrame up to a specified gameweek.

    Parameters:
    - gw_data_frames: Dictionary of DataFrames for each gameweek.
    - up_to_gw: Integer, the last gameweek to include in the feature calculation.

    Returns:
    - DataFrame with defender features calculated up to the specified gameweek.
    """
    # Combine DataFrames up to the specified gameweek
    combined_df = pd.concat([df for gw, df in gw_data_frames.items() if gw <= up_to_gw], ignore_index=True)

    # Filter for defenders
    df_defender = combined_df.loc[combined_df['position'] == 'DEF']

    # Group by player name
    defender_grouped = df_defender.groupby('name')

    # Initialize the features DataFrame
    defender_features = pd.DataFrame(index=defender_grouped.indices.keys())

    # Calculate features relevant to defenders
    defender_features['minutes'] = defender_grouped['minutes'].mean()
    defender_features['clean_sheets'] = defender_grouped['clean_sheets'].sum()
    defender_features['goals_conceded'] = defender_grouped['goals_conceded'].mean()
    defender_features['goals_scored'] = defender_grouped['goals_scored'].sum()  # Points for goals scored by defenders
    defender_features['assists'] = defender_grouped['assists'].sum()
    defender_features['bonus'] = defender_grouped['bonus'].mean()
    defender_features['yellow_cards'] = defender_grouped['yellow_cards'].sum()  # Points deduction for yellow cards
    defender_features['red_cards'] = defender_grouped['red_cards'].sum()  # Points deduction for red cards
    defender_features['own_goals'] = defender_grouped['own_goals'].sum()  # Points deduction for own goals

    # Calculate recent form (total points in the last 5 games)
    recent_form = []
    for name, group in defender_grouped:
        group = group.sort_values(by='gw', ascending=False)
        recent_form.append(group['total_points'].head(5).sum())
    defender_features['recent_form'] = recent_form

    return defender_features


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
    
    defender_features = create_defender_features(gw_data_frames, target_gw - 1)
    
    # Load target gameweek data
    gw_target_data = gw_data_frames[target_gw]
    gw_target_defender = gw_target_data.loc[gw_target_data['position'] == 'DEF']
    
    # Prepare target variable
    y = gw_target_defender[['name', 'total_points']].set_index('name').sort_index()

    common_indices = defender_features.index.intersection(y.index)
    x_filtered = defender_features.loc[common_indices]
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