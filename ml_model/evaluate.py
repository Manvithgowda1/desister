"""
evaluate.py - Model evaluation: classification report, confusion matrix, AUC-ROC.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import train_model, load_saved_model_and_test_data

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")


def evaluate_model(model, X_test, y_test):
    """
    Run full evaluation suite and save results.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 1. Classification Report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        y_test, y_pred,
        target_names=["No Earthquake", "Earthquake"],
        digits=4
    )
    print(report)
    
    # Save report to file
    report_path = os.path.join(SAVE_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    
    # 2. Confusion Matrix
    print("CONFUSION MATRIX")
    print("-" * 40)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Earthquake", "Earthquake"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Earthquake Prediction - Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"   Saved to {cm_path}")
    
    # 3. AUC-ROC
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nAUC-ROC Score: {auc:.4f}")
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#ef4444", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve - Earthquake Prediction", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(SAVE_DIR, "roc_curve.png")
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"   Saved to {roc_path}")
    
    # 4. Feature Importance (XGBoost built-in)
    print("\nFEATURE IMPORTANCE (Gain)")
    print("-" * 40)
    importance = model.feature_importances_
    feat_names = X_test.columns.tolist()
    sorted_idx = np.argsort(importance)[::-1]
    
    for i in sorted_idx:
        print(f"   {feat_names[i]:35s}  {importance[i]:.4f}")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_asc = np.argsort(importance)
    ax.barh(
        [feat_names[i] for i in sorted_asc],
        importance[sorted_asc],
        color="#3b82f6"
    )
    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_title("XGBoost Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fi_path = os.path.join(SAVE_DIR, "feature_importance.png")
    fig.savefig(fi_path, dpi=150)
    plt.close(fig)
    print(f"   Saved to {fi_path}")
    
    print(f"\n{'=' * 60}")
    print(f"All evaluation artifacts saved to {SAVE_DIR}")
    print(f"{'=' * 60}")
    
    return {"auc_roc": auc, "classification_report": report, "confusion_matrix": cm}


if __name__ == "__main__":
    print("Loading model and test data for evaluation...")
    model, X_test, y_test, _ = load_saved_model_and_test_data()
    results = evaluate_model(model, X_test, y_test)
