"""
train.py - XGBoost model training with SMOTE class imbalance handling.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import generate_dataset, save_dataset
from feature_engineering import (
    engineer_features, scale_features, get_all_feature_columns, TARGET_COLUMN
)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")


def train_model(use_smote=True):
    """
    Full training pipeline:
      1. Generate / load dataset
      2. Engineer features
      3. Split train/test (stratified 80/20)
      4. Handle class imbalance (SMOTE + scale_pos_weight)
      5. Train XGBClassifier
      6. Save model and scaler
    
    Returns:
        model, X_test_scaled, y_test, feature_columns
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Dataset
    print("=" * 60)
    print("Step 1/5: Generating dataset...")
    print("=" * 60)
    df = generate_dataset(n_samples=5000)
    save_dataset(df, SAVE_DIR)
    
    # 2. Feature engineering
    print("\nStep 2/5: Engineering features...")
    df = engineer_features(df)
    
    feature_cols = get_all_feature_columns()
    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples:  {len(X)}")
    print(f"   Positive: {y.sum()} ({y.mean():.1%})")
    print(f"   Negative: {(1 - y).sum()} ({(1 - y).mean():.1%})")
    
    # 3. Stratified split
    print("\nStep 3/5: Splitting train/test (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # 4. Class imbalance handling
    print("\nStep 4/5: Handling class imbalance...")
    
    # Calculate scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"   scale_pos_weight = {scale_pos_weight:.2f}")
    
    # Optional SMOTE oversampling
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"   SMOTE applied: {len(X_train)} -> {len(X_train_resampled)} samples")
            X_train = pd.DataFrame(X_train_resampled, columns=feature_cols)
            y_train = pd.Series(y_train_resampled)
        except ImportError:
            print("   Warning: imbalanced-learn not installed, skipping SMOTE.")
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, SAVE_DIR)
    
    # 5. Train XGBoost
    print("\nStep 5/5: Training XGBClassifier...")
    model = XGBClassifier(
        max_depth=6,
        n_estimators=200,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # Save model
    model_path = os.path.join(SAVE_DIR, "earthquake_model.json")
    model.save_model(model_path)
    print(f"\n   Model saved to {model_path}")
    
    # Also save with joblib for easy loading with scaler
    joblib_path = os.path.join(SAVE_DIR, "earthquake_model.pkl")
    joblib.dump(model, joblib_path)
    
    # Save feature columns list
    cols_path = os.path.join(SAVE_DIR, "feature_columns.txt")
    with open(cols_path, "w") as f:
        f.write("\n".join(feature_cols))
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return model, X_test_scaled, y_test, feature_cols


def load_saved_model_and_test_data():
    """
    Loads the pre-trained model, scaler, and returns the test dataset scaled, y_test, and feature_cols.
    """
    model_pkl_path = os.path.join(SAVE_DIR, "earthquake_model.pkl")
    scaler_pkl_path = os.path.join(SAVE_DIR, "scaler.pkl")
    dataset_csv_path = os.path.join(SAVE_DIR, "earthquake_dataset.csv")
    
    if not (os.path.exists(model_pkl_path) and os.path.exists(scaler_pkl_path) and os.path.exists(dataset_csv_path)):
        print("Pre-trained model or data not found. Training model from scratch...")
        return train_model(use_smote=True)
    
    print("Loading pre-trained model and scaler...")
    model = joblib.load(model_pkl_path)
    scaler = joblib.load(scaler_pkl_path)
    
    print("Loading existing dataset and preparing features...")
    df = pd.read_csv(dataset_csv_path)
    df = engineer_features(df)
    
    feature_cols = get_all_feature_columns()
    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
    return model, X_test_scaled, y_test, feature_cols


if __name__ == "__main__":
    model, X_test, y_test, feature_cols = train_model(use_smote=True)
    
    # Quick accuracy preview
    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_test)
    print(f"\nQuick test accuracy: {accuracy_score(y_test, y_pred):.4f}")

