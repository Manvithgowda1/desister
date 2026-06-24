"""
predict_api.py - Flask REST API for earthquake risk prediction.
Loads the trained XGBoost model and scaler, serves predictions via HTTP.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import engineer_features, get_all_feature_columns

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")

app = Flask(__name__)

# Global model and scaler (loaded once at startup)
_model = None
_scaler = None
_shap_explainer = None


def load_model():
    """Load the trained model and scaler from disk."""
    global _model, _scaler
    
    model_path = os.path.join(SAVE_DIR, "earthquake_model.pkl")
    scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Run 'python train.py' first to train and save the model."
        )
    
    _model = joblib.load(model_path)
    
    if os.path.exists(scaler_path):
        _scaler = joblib.load(scaler_path)
    else:
        print("Warning: Scaler not found, predictions will use unscaled features.")
        _scaler = None
    
    print("Model and scaler loaded successfully.")


def get_shap_explainer():
    """Lazily initialize and return a SHAP TreeExplainer."""
    global _shap_explainer
    if _shap_explainer is None:
        try:
            import shap
            _shap_explainer = shap.TreeExplainer(_model)
        except ImportError:
            print("Warning: SHAP not installed, explanations will not be available.")
    return _shap_explainer


def classify_risk(probability):
    """Convert probability to risk category and score."""
    score = int(round(probability * 100))
    if score <= 33:
        category = "Low"
    elif score <= 66:
        category = "Moderate"
    else:
        category = "High"
    return category, score


@app.route("/api/predict-earthquake", methods=["POST"])
def predict_earthquake():
    """
    Predict earthquake risk for a given location.
    
    Expected JSON body:
    {
        "latitude": 28.6,
        "longitude": 77.2,
        "seismic_zone": 4,
        "historical_eq_frequency": 12,
        "avg_historical_magnitude": 5.2,
        "fault_proximity_km": 15.0,
        "population_density": 11320
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400
        
        # Validate required fields
        required = [
            "latitude", "longitude", "seismic_zone",
            "historical_eq_frequency", "avg_historical_magnitude",
            "fault_proximity_km", "population_density"
        ]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400
        
        # Build single-row DataFrame
        input_df = pd.DataFrame([{
            "latitude": float(data["latitude"]),
            "longitude": float(data["longitude"]),
            "seismic_zone": int(data["seismic_zone"]),
            "historical_eq_frequency": int(data["historical_eq_frequency"]),
            "avg_historical_magnitude": float(data["avg_historical_magnitude"]),
            "fault_proximity_km": float(data["fault_proximity_km"]),
            "population_density": float(data["population_density"]),
        }])
        
        # Feature engineering
        input_df = engineer_features(input_df)
        feature_cols = get_all_feature_columns()
        X = input_df[feature_cols]
        
        # Scale
        if _scaler is not None:
            X_scaled = pd.DataFrame(
                _scaler.transform(X),
                columns=feature_cols
            )
        else:
            X_scaled = X
        
        # Predict
        probability = float(_model.predict_proba(X_scaled)[:, 1][0])
        risk_category, risk_score = classify_risk(probability)
        
        # SHAP explanations (optional)
        top_factors = []
        explainer = get_shap_explainer()
        if explainer is not None:
            try:
                shap_values = explainer.shap_values(X_scaled)
                shap_row = shap_values[0]
                
                # Sort features by absolute SHAP impact
                sorted_idx = np.argsort(np.abs(shap_row))[::-1]
                for i in sorted_idx[:5]:  # Top 5 factors
                    top_factors.append({
                        "feature": feature_cols[i],
                        "impact": round(float(shap_row[i]), 4)
                    })
            except Exception as e:
                print(f"SHAP explanation error: {e}")
        
        response = {
            "probability": round(probability, 4),
            "risk_category": risk_category,
            "risk_score": risk_score,
            "top_factors": top_factors,
            "input_received": data
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
        "scaler_loaded": _scaler is not None
    })


def start_api(host="127.0.0.1", port=5001):
    """Start the Flask prediction API server."""
    load_model()
    print(f"\nEarthquake Prediction API running at http://{host}:{port}")
    print(f"   POST /api/predict-earthquake")
    print(f"   GET  /api/health\n")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    start_api()
