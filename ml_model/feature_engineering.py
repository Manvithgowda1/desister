"""
feature_engineering.py - Feature transformations and scaling for earthquake prediction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os


FEATURE_COLUMNS = [
    "latitude", "longitude", "seismic_zone",
    "historical_eq_frequency", "avg_historical_magnitude",
    "fault_proximity_km", "population_density"
]

ENGINEERED_COLUMNS = [
    "zone_magnitude_interaction",
    "log_fault_proximity",
    "freq_per_zone",
    "high_density_zone"
]

TARGET_COLUMN = "earthquake_occurred"


def engineer_features(df):
    """
    Create derived features from the raw input dataframe.
    
    New columns:
        zone_magnitude_interaction: seismic_zone * avg_historical_magnitude
        log_fault_proximity: log(1 + fault_proximity_km)
        freq_per_zone: historical_eq_frequency / seismic_zone
        high_density_zone: 1 if population_density > 1000 AND seismic_zone >= 4
    """
    df = df.copy()
    
    # 1. Zone-Magnitude interaction
    df["zone_magnitude_interaction"] = df["seismic_zone"] * df["avg_historical_magnitude"]
    
    # 2. Log-transformed fault proximity (diminishing returns)
    df["log_fault_proximity"] = np.log1p(df["fault_proximity_km"])
    
    # 3. Frequency normalised by zone severity
    df["freq_per_zone"] = df["historical_eq_frequency"] / df["seismic_zone"]
    
    # 4. High-density high-zone vulnerability flag
    df["high_density_zone"] = (
        (df["population_density"] > 1000) & (df["seismic_zone"] >= 4)
    ).astype(int)
    
    return df


def get_all_feature_columns():
    """Return the full list of feature columns after engineering."""
    return FEATURE_COLUMNS + ENGINEERED_COLUMNS


def scale_features(X_train, X_test, save_dir=None):
    """
    Apply StandardScaler to continuous features.
    Fits on train, transforms both train and test.
    Saves the scaler to disk if save_dir is provided.
    
    Returns:
        X_train_scaled (DataFrame), X_test_scaled (DataFrame), scaler
    """
    scaler = StandardScaler()
    
    all_cols = get_all_feature_columns()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[all_cols]),
        columns=all_cols,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[all_cols]),
        columns=all_cols,
        index=X_test.index
    )
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
    
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    from dataset import generate_dataset
    
    df = generate_dataset(500)
    df = engineer_features(df)
    
    print("Engineered features sample:")
    print(df[get_all_feature_columns()].head(10).to_string())
    print(f"\nAll feature columns: {get_all_feature_columns()}")
