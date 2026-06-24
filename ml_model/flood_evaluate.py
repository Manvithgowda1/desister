"""
flood_evaluate.py - Model evaluation module for flood prediction
Comprehensive evaluation metrics including accuracy, precision, recall, F1, ROC-AUC, and risk categorization.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


class FloodModelEvaluator:
    """Evaluator class for flood prediction model."""
    
    def __init__(self):
        self.metrics = {}
        self.confusion_matrix = None
        self.risk_categories = None
        
    def evaluate(self, y_true, y_pred, y_pred_proba=None):
        """
        Comprehensive evaluation of the model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("Evaluating model performance...")
        
        # Basic metrics
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(y_true, y_pred)
        self.metrics['recall'] = recall_score(y_true, y_pred)
        self.metrics['f1_score'] = f1_score(y_true, y_pred)
        
        # Confusion matrix
        self.confusion_matrix = confusion_matrix(y_true, y_pred)
        
        # ROC-AUC if probabilities available
        if y_pred_proba is not None:
            self.metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Additional metrics
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        self.metrics['true_negatives'] = int(tn)
        self.metrics['false_positives'] = int(fp)
        self.metrics['false_negatives'] = int(fn)
        self.metrics['true_positives'] = int(tp)
        
        # Calculate risk categories
        self.risk_categories = self._categorize_risk(y_pred_proba if y_pred_proba is not None else y_pred)
        
        return self.metrics
    
    def _categorize_risk(self, predictions):
        """
        Categorize predictions into risk levels.
        
        Args:
            predictions: Either probabilities or binary predictions
            
        Returns:
            Series with risk categories
        """
        if len(predictions.shape) == 1 and predictions.dtype == float:
            # Probabilities
            categories = pd.cut(
                predictions,
                bins=[0, 0.25, 0.5, 0.75, 1.0],
                labels=['Low', 'Moderate', 'High', 'Very High']
            )
        else:
            # Binary predictions
            categories = pd.Series(predictions).map({
                0: 'Low',
                1: 'High'
            })
        
        return categories
    
    def print_report(self):
        """Print a comprehensive evaluation report."""
        print("\n" + "="*60)
        print("FLOOD PREDICTION MODEL EVALUATION REPORT")
        print("="*60)
        
        print("\n[Performance Metrics]")
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1 Score:  {self.metrics['f1_score']:.4f}")
        
        if 'roc_auc' in self.metrics:
            print(f"ROC-AUC:   {self.metrics['roc_auc']:.4f}")
        
        print("\n[Confusion Matrix]")
        print(f"True Negatives:  {self.metrics['true_negatives']}")
        print(f"False Positives: {self.metrics['false_positives']}")
        print(f"False Negatives: {self.metrics['false_negatives']}")
        print(f"True Positives:  {self.metrics['true_positives']}")
        
        print("\n[Confusion Matrix Visualization]")
        print(self.confusion_matrix)
        
        if self.risk_categories is not None:
            print("\n[Risk Category Distribution]")
            print(self.risk_categories.value_counts().sort_index())
        
        print("\n" + "="*60)
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Plot the confusion matrix.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Flood', 'Flood'],
            yticklabels=['No Flood', 'Flood']
        )
        plt.title('Confusion Matrix - Flood Prediction')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot (optional)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {self.metrics["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Flood Prediction')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot (optional)
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Flood Prediction')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_report(self, output_dir):
        """
        Save evaluation report to file.
        
        Args:
            output_dir: Directory to save the report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f'flood_evaluation_report_{timestamp}.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("FLOOD PREDICTION MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("[Performance Metrics]\n")
            for metric, value in self.metrics.items():
                if isinstance(value, float):
                    f.write(f"{metric}: {value:.4f}\n")
                else:
                    f.write(f"{metric}: {value}\n")
            
            f.write("\n[Confusion Matrix]\n")
            f.write(str(self.confusion_matrix) + "\n")
            
            if self.risk_categories is not None:
                f.write("\n[Risk Category Distribution]\n")
                f.write(str(self.risk_categories.value_counts().sort_index()) + "\n")
        
        print(f"Evaluation report saved to {report_path}")
        return report_path


def evaluate_model(model, X_test, y_test, preprocessor, output_dir=None):
    """
    Complete evaluation pipeline for a trained model.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
        preprocessor: Fitted preprocessor
        output_dir: Directory to save evaluation artifacts (optional)
        
    Returns:
        evaluator instance with metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    evaluator = FloodModelEvaluator()
    evaluator.evaluate(y_test, y_pred, y_pred_proba)
    
    # Print report
    evaluator.print_report()
    
    # Generate plots if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrix
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        evaluator.plot_confusion_matrix(cm_path)
        
        # Plot ROC curve
        roc_path = os.path.join(output_dir, 'roc_curve.png')
        evaluator.plot_roc_curve(y_test, y_pred_proba, roc_path)
        
        # Plot Precision-Recall curve
        pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
        evaluator.plot_precision_recall_curve(y_test, y_pred_proba, pr_path)
        
        # Save text report
        evaluator.save_report(output_dir)
    
    return evaluator


def load_saved_model_and_data(output_dir):
    from flood_train import FloodModelTrainer
    from flood_preprocessing import prepare_data
    
    # Check if dataset is in output_dir, else look in parent directory's saved_models
    model_path = os.path.join(output_dir, 'flood_xgboost_model.json')
    preprocessor_path = os.path.join(output_dir, 'flood_preprocessor.pkl')
    dataset_path = os.path.join(output_dir, 'flood_dataset.csv')
    
    # Fallback to parent dir if output_dir is 'evaluation' subdirectory
    if not os.path.exists(model_path):
        parent_dir = os.path.dirname(output_dir)
        model_path = os.path.join(parent_dir, 'flood_xgboost_model.json')
        preprocessor_path = os.path.join(parent_dir, 'flood_preprocessor.pkl')
        dataset_path = os.path.join(parent_dir, 'flood_dataset.csv')

    if not (os.path.exists(model_path) and os.path.exists(preprocessor_path) and os.path.exists(dataset_path)):
        print("Pre-trained flood model or data not found. Training model from scratch...")
        from flood_train import train_full_pipeline
        return train_full_pipeline(n_samples=5000, test_size=0.2, use_grid_search=False)
        
    print("Loading pre-trained flood model...")
    trainer = FloodModelTrainer.load_model(model_path, preprocessor_path)
    
    print("Loading existing flood dataset and preparing test data...")
    df = pd.read_csv(dataset_path)
    
    # Process using prepare_data to get same splits and transformations
    _, X_test_proc, _, y_test, preprocessor = prepare_data(df, test_size=0.2, random_state=42)
    
    return trainer, X_test_proc, y_test, preprocessor


if __name__ == "__main__":
    from flood_preprocessing import FloodPreprocessor
    
    print("Running evaluation pipeline...")
    
    output_dir = os.path.join(os.path.dirname(__file__), 'saved_models', 'evaluation')
    base_saved_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    
    # Load model and test data
    trainer, X_test, y_test, preprocessor = load_saved_model_and_data(base_saved_dir)
    
    # Evaluate model
    evaluator = evaluate_model(
        trainer.model,
        X_test,
        y_test,
        preprocessor,
        output_dir
    )
    
    print("\nEvaluation completed successfully!")

