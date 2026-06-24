"""
flood_train.py - XGBoost training pipeline for flood prediction
Trains an XGBoost classifier for flood risk prediction with hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from datetime import datetime
import json

from flood_dataset import generate_dataset, save_dataset
from flood_preprocessing import prepare_data, FloodPreprocessor


class FloodModelTrainer:
    """Trainer class for XGBoost flood prediction model."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.preprocessor = None
        self.best_params = None
        self.training_history = {}
        
    def train(self, X_train, y_train, X_val=None, y_val=None, use_grid_search=True):
        """
        Train the XGBoost model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            use_grid_search: Whether to perform hyperparameter tuning
            
        Returns:
            trained model
        """
        print("Starting XGBoost training...")
        
        if use_grid_search:
            print("Performing hyperparameter tuning with GridSearchCV...")
            self.model = self._grid_search(X_train, y_train)
        else:
            print("Training with default parameters...")
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
            self.model.fit(X_train, y_train)
            self.best_params = self.model.get_params()
        
        # Store training history
        self.training_history['training_samples'] = len(X_train)
        self.training_history['positive_class_ratio'] = y_train.mean()
        self.training_history['best_params'] = self.best_params
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            self.training_history['validation_accuracy'] = val_score
            print(f"Validation accuracy: {val_score:.4f}")
        
        print("Training completed successfully!")
        return self.model
    
    def _grid_search(self, X_train, y_train):
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Best XGBoost model from grid search
        """
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5]
        }
        
        # Initialize base model
        xgb_base = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=xgb_base,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        print(f"\nBest parameters found: {self.best_params}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance from the trained model.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, output_dir, model_name='flood_xgboost_model'):
        """
        Save the trained model and preprocessor to disk.
        
        Args:
            output_dir: Directory to save the model
            model_name: Name for the model file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, f'{model_name}.json')
        self.model.save_model(model_path)
        print(f"Model saved to {model_path}")
        
        # Save training history
        history_path = os.path.join(output_dir, f'{model_name}_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy types to serializable types
            history_serializable = {}
            for key, value in self.training_history.items():
                if isinstance(value, (np.integer, np.floating)):
                    history_serializable[key] = float(value)
                elif isinstance(value, dict):
                    history_serializable[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                                 for k, v in value.items()}
                else:
                    history_serializable[key] = value
            json.dump(history_serializable, f, indent=2)
        print(f"Training history saved to {history_path}")
        
        # Save preprocessor if available
        if self.preprocessor is not None:
            self.preprocessor.save(output_dir)
        
        return model_path
    
    @classmethod
    def load_model(cls, model_path, preprocessor_path=None):
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the model file
            preprocessor_path: Path to the preprocessor file (optional)
            
        Returns:
            FloodModelTrainer instance with loaded model
        """
        trainer = cls()
        trainer.model = xgb.XGBClassifier()
        trainer.model.load_model(model_path)
        
        if preprocessor_path:
            trainer.preprocessor = FloodPreprocessor.load(os.path.dirname(preprocessor_path))
        
        # Load training history if available
        history_path = model_path.replace('.json', '_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                trainer.training_history = json.load(f)
        
        print(f"Model loaded from {model_path}")
        return trainer


def train_full_pipeline(n_samples=5000, test_size=0.2, use_grid_search=True):
    """
    Complete training pipeline from data generation to model saving.
    
    Args:
        n_samples: Number of samples to generate
        test_size: Proportion of data for testing
        use_grid_search: Whether to use hyperparameter tuning
        
    Returns:
        trainer, X_test, y_test, preprocessor
    """
    print("="*60)
    print("FLOOD PREDICTION MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Generate dataset
    print("\n[Step 1/5] Generating synthetic flood dataset...")
    df = generate_dataset(n_samples=n_samples)
    save_dataset(df)
    print(f"Dataset generated with {df.shape[0]} samples")
    
    # Step 2: Prepare data
    print("\n[Step 2/5] Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df, test_size=test_size)
    
    # Step 3: Train model
    print("\n[Step 3/5] Training XGBoost model...")
    trainer = FloodModelTrainer()
    trainer.preprocessor = preprocessor
    trainer.train(X_train, y_train, X_test, y_test, use_grid_search=use_grid_search)
    
    # Step 4: Feature importance
    print("\n[Step 4/5] Computing feature importance...")
    feature_names = preprocessor.get_feature_names()
    importance_df = trainer.get_feature_importance(feature_names)
    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))
    
    # Step 5: Save model
    print("\n[Step 5/5] Saving model and artifacts...")
    output_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    trainer.save_model(output_dir)
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return trainer, X_test, y_test, preprocessor


if __name__ == "__main__":
    # Train the model
    trainer, X_test, y_test, preprocessor = train_full_pipeline(
        n_samples=5000, 
        test_size=0.2, 
        use_grid_search=True
    )
    
    # Quick evaluation
    print("\nQuick evaluation on test set:")
    y_pred = trainer.model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
