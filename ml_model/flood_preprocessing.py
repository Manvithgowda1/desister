"""
flood_preprocessing.py - Data preprocessing pipeline for flood prediction
Handles feature engineering, encoding, and scaling for flood risk prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os


class FloodPreprocessor:
    """Preprocessing pipeline for flood prediction features."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.numeric_features = [
            'rainfall_mm', 
            'river_water_level_m', 
            'soil_moisture', 
            'elevation_m', 
            'historical_flood_events'
        ]
        self.categorical_features = ['district', 'state', 'basin']
        
    def fit(self, df):
        """
        Fit the preprocessor on the training data.
        
        Args:
            df: pandas DataFrame with raw features
        """
        # Store feature columns
        self.feature_columns = df.columns.tolist()
        
        # Fit label encoders for categorical features
        for col in self.categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Fit scaler on numeric features
        numeric_data = df[self.numeric_features].values
        self.scaler.fit(numeric_data)
        
        return self
    
    def transform(self, df):
        """
        Transform the data using fitted preprocessor.
        
        Args:
            df: pandas DataFrame with raw features
            
        Returns:
            numpy array of preprocessed features
        """
        df_transformed = df.copy()
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in df_transformed.columns and col in self.label_encoders:
                df_transformed[col] = self.label_encoders[col].transform(
                    df_transformed[col].astype(str)
                )
        
        # Scale numeric features
        df_transformed[self.numeric_features] = self.scaler.transform(
            df_transformed[self.numeric_features]
        )
        
        # Return all features as numpy array
        feature_cols = self.categorical_features + self.numeric_features
        return df_transformed[feature_cols].values
    
    def fit_transform(self, df):
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def get_feature_names(self):
        """Return the names of features after preprocessing."""
        return self.categorical_features + self.numeric_features
    
    def save(self, output_dir):
        """Save the preprocessor to disk."""
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self, os.path.join(output_dir, 'flood_preprocessor.pkl'))
        print(f"Preprocessor saved to {output_dir}")
    
    @classmethod
    def load(cls, input_dir):
        """Load a preprocessor from disk."""
        preprocessor = joblib.load(os.path.join(input_dir, 'flood_preprocessor.pkl'))
        print(f"Preprocessor loaded from {input_dir}")
        return preprocessor


def prepare_data(df, test_size=0.2, random_state=42):
    """
    Prepare data for training by splitting and preprocessing.
    
    Args:
        df: pandas DataFrame with features and target
        test_size: proportion of data for testing
        random_state: random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    # Separate features and target
    feature_cols = ['district', 'state', 'basin', 'rainfall_mm', 
                    'river_water_level_m', 'soil_moisture', 'elevation_m', 
                    'historical_flood_events']
    X = df[feature_cols].copy()
    y = df['flood_occurred'].copy()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize and fit preprocessor
    preprocessor = FloodPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Training set size: {X_train_processed.shape}")
    print(f"Test set size: {X_test_processed.shape}")
    print(f"Training set flood rate: {y_train.mean():.2%}")
    print(f"Test set flood rate: {y_test.mean():.2%}")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def engineer_features(df):
    """
    Engineer additional features for flood prediction.
    
    Args:
        df: pandas DataFrame with raw features
        
    Returns:
        pandas DataFrame with engineered features
    """
    df = df.copy()
    
    # Rainfall intensity categories
    df['rainfall_category'] = pd.cut(
        df['rainfall_mm'], 
        bins=[0, 500, 1000, 1500, 2000, float('inf')],
        labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    )
    
    # River level danger zone
    df['river_danger_zone'] = (df['river_water_level_m'] > 15).astype(int)
    
    # Soil moisture saturation
    df['soil_saturation'] = pd.cut(
        df['soil_moisture'],
        bins=[0, 0.3, 0.5, 0.7, 0.85, 1.0],
        labels=['Dry', 'Low', 'Medium', 'High', 'Saturated']
    )
    
    # Elevation risk (lower elevation = higher risk)
    df['elevation_risk'] = pd.cut(
        df['elevation_m'],
        bins=[0, 50, 100, 200, 500, float('inf')],
        labels=['Very High', 'High', 'Medium', 'Low', 'Very Low']
    )
    
    # Historical flood frequency
    df['historical_risk'] = pd.cut(
        df['historical_flood_events'],
        bins=[0, 5, 10, 15, 20, float('inf')],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Combined risk score (simple heuristic)
    df['risk_score'] = (
        (df['rainfall_mm'] / 3000) * 0.3 +
        (df['river_water_level_m'] / 20) * 0.25 +
        df['soil_moisture'] * 0.2 +
        (1 - df['elevation_m'] / 1000) * 0.15 +
        (df['historical_flood_events'] / 20) * 0.1
    )
    
    return df


if __name__ == "__main__":
    from flood_dataset import generate_dataset
    
    print("Testing preprocessing pipeline...")
    df = generate_dataset(n_samples=1000)
    
    print("\nOriginal data shape:", df.shape)
    print("Original columns:", df.columns.tolist())
    
    # Engineer features
    df_engineered = engineer_features(df)
    print("\nEngineered data shape:", df_engineered.shape)
    print("Engineered columns:", df_engineered.columns.tolist())
    
    # Prepare data for training
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df_engineered)
    
    print("\nPreprocessed training data shape:", X_train.shape)
    print("Feature names:", preprocessor.get_feature_names())
