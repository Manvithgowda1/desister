"""
explain.py - SHAP explainability for the earthquake prediction model.
Generates summary, force, and dependence plots.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import train_model, load_saved_model_and_test_data

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")


def generate_shap_explanations(model, X_test, y_test):
    """
    Generate SHAP explanations for the trained XGBoost model.
    Saves summary, force, and dependence plots as PNG.
    """
    import shap
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)
    
    # Create TreeExplainer (optimised for tree-based models)
    print("\nInitializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # 1. Summary Plot (Beeswarm) - Global feature importance
    print("Generating Summary (Beeswarm) Plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot - Earthquake Risk Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    summary_path = os.path.join(SAVE_DIR, "shap_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"   Saved to {summary_path}")
    
    # 2. Bar Plot - Mean absolute SHAP values
    print("Generating SHAP Bar Plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    bar_path = os.path.join(SAVE_DIR, "shap_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"   Saved to {bar_path}")
    
    # 3. Force Plot - Single prediction explanation (first positive sample)
    print("Generating Force Plot for a sample prediction...")
    positive_indices = np.where(y_test.values == 1)[0]
    if len(positive_indices) > 0:
        sample_idx = positive_indices[0]
        
        # Save as static matplotlib figure
        fig = plt.figure(figsize=(14, 4))
        shap.force_plot(
            explainer.expected_value,
            shap_values[sample_idx, :],
            X_test.iloc[sample_idx, :],
            matplotlib=True,
            show=False
        )
        plt.title("SHAP Force Plot (Sample Earthquake Prediction)", fontsize=12, fontweight="bold")
        force_path = os.path.join(SAVE_DIR, "shap_force.png")
        plt.savefig(force_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"   Saved to {force_path}")
    else:
        print("   No positive samples found for force plot.")
    
    # 4. Dependence Plots - Top 3 most important features
    print("Generating Dependence Plots for top features...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[::-1][:3]
    feature_names = X_test.columns.tolist()
    
    for rank, feat_idx in enumerate(top_features_idx):
        feat_name = feature_names[feat_idx]
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            feat_idx, shap_values, X_test,
            show=False, ax=ax
        )
        ax.set_title(f"SHAP Dependence: {feat_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        dep_path = os.path.join(SAVE_DIR, f"shap_dependence_{feat_name}.png")
        fig.savefig(dep_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   [{rank + 1}] {feat_name} -> {dep_path}")
    
    # 5. Print top SHAP values summary
    print("\n" + "-" * 50)
    print("SHAP MEAN ABSOLUTE VALUES (Global Importance)")
    print("-" * 50)
    for idx in np.argsort(mean_abs_shap)[::-1]:
        print(f"   {feature_names[idx]:35s}  {mean_abs_shap[idx]:.4f}")
    
    print(f"\n{'=' * 60}")
    print(f"All SHAP plots saved to {SAVE_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    print("Loading model and test data for SHAP analysis...")
    model, X_test, y_test, _ = load_saved_model_and_test_data()
    generate_shap_explanations(model, X_test, y_test)
