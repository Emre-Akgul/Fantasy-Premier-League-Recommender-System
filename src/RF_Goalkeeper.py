import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def create_goalkeeper_features(gw_data_frames, up_to_gw):
    """
    Create goalkeeper features DataFrame up to a specified gameweek.

    Parameters:
    - gw_data_frames: Dictionary of DataFrames for each gameweek.
    - up_to_gw: Integer, the last gameweek to include in the feature calculation.

    Returns:
    - DataFrame with goalkeeper features calculated up to the specified gameweek.
    """
    # Combine DataFrames up to the specified gameweek
    combined_df = pd.concat([df for gw, df in gw_data_frames.items() if gw <= up_to_gw], ignore_index=True)

    # Filter for goalkeepers
    df_goalkeeper = combined_df.loc[combined_df['position'] == 'GK']

    # Group by player name
    goalkeeper_grouped = df_goalkeeper.groupby('name')

    # Initialize the features DataFrame
    goalkeeper_features = pd.DataFrame(index=goalkeeper_grouped.indices.keys())

    # Calculate features
    goalkeeper_features['minutes'] = goalkeeper_grouped['minutes'].mean()
    goalkeeper_features['clean_sheets'] = goalkeeper_grouped['clean_sheets'].sum()
    goalkeeper_features['goals_conceded'] = goalkeeper_grouped['goals_conceded'].mean()
    goalkeeper_features['saves'] = goalkeeper_grouped['saves'].sum()
    goalkeeper_features['penalties_saved'] = goalkeeper_grouped['penalties_saved'].sum()
    goalkeeper_features['bonus'] = goalkeeper_grouped['bonus'].mean()

    # Calculate recent form (total points in the last 5 games)
    recent_form = []
    for name, group in goalkeeper_grouped:
        group = group.sort_values(by='gw', ascending=False)
        recent_form.append(group['total_points'].head(5).sum())
    goalkeeper_features['recent_form'] = recent_form

    return goalkeeper_features

def train_and_evaluate_rf(gw_data_frames, target_gw):
    """
    Train a Random Forest model with data up to a specified gameweek and evaluate it using the target gameweek.
    Automatically excludes goalkeepers not present in the feature set.

    Parameters:
    - gw_data_frames: Dictionary of DataFrames for each gameweek.
    - target_gw: Integer, the gameweek to predict and evaluate the model.

    Returns:
    - mse: Mean Squared Error of the model predictions for the target gameweek.
    """
    
    # Assume create_goalkeeper_features function is defined as before
    goalkeeper_features = create_goalkeeper_features(gw_data_frames, target_gw - 1)
    
    # Load target gameweek data
    gw_target_data = gw_data_frames[target_gw]
    gw_target_goalkeeper = gw_target_data.loc[gw_target_data['position'] == 'GK']
    
    # Prepare target variable
    y = gw_target_goalkeeper[['name', 'total_points']].set_index('name').sort_index()

    # Ensure we only consider goalkeepers present in both the features and target sets
    common_indices = goalkeeper_features.index.intersection(y.index)
    x_filtered = goalkeeper_features.loc[common_indices]
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
