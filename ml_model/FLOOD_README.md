# Flood Prediction System for India

XGBoost-based flood risk prediction system for Indian districts with comprehensive ML pipeline and FastAPI endpoint.

## Features

- **Input Features**: Rainfall, River Water Level, Soil Moisture, Elevation, Historical Flood Events, District/State/Basin
- **Output**: Flood probability percentage and risk category (Low, Moderate, High, Very High)
- **Model**: XGBoost Classifier with hyperparameter tuning
- **API**: FastAPI REST endpoint for real-time predictions

## Installation

```bash
cd ml_model
pip install -r flood_requirements.txt
```

## Quick Start

### 1. Generate Dataset

```bash
python flood_dataset.py
```

This generates a synthetic flood dataset with 5000 samples across 18 Indian districts in major river basins (Ganga, Brahmaputra, Godavari, Krishna, Mahanadi, Kaveri, Coastal, Arid regions).

### 2. Train Model

```bash
python flood_train.py
```

This runs the complete training pipeline:
- Data generation
- Preprocessing (encoding, scaling)
- XGBoost training with GridSearchCV
- Feature importance analysis
- Model saving to `saved_models/`

### 3. Evaluate Model

```bash
python flood_evaluate.py
```

This evaluates the trained model and generates:
- Performance metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix visualization
- ROC curve
- Precision-Recall curve
- Evaluation report

### 4. Start API Server

```bash
python flood_api.py
```

The API will be available at `http://localhost:8000`

## API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "district": "Patna",
    "state": "Bihar",
    "basin": "Ganga",
    "rainfall_mm": 1200.0,
    "river_water_level_m": 15.0,
    "soil_moisture": 0.65,
    "elevation_m": 55.0,
    "historical_flood_events": 5
  }'
```

Response:
```json
{
  "district": "Patna",
  "state": "Bihar",
  "flood_probability": 78.45,
  "risk_category": "Very High",
  "prediction_timestamp": "2024-01-15T10:30:00",
  "model_version": "1.0.0"
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {
        "district": "Patna",
        "state": "Bihar",
        "basin": "Ganga",
        "rainfall_mm": 1200.0,
        "river_water_level_m": 15.0,
        "soil_moisture": 0.65,
        "elevation_m": 55.0,
        "historical_flood_events": 5
      },
      {
        "district": "Jaipur",
        "state": "Rajasthan",
        "basin": "Arid",
        "rainfall_mm": 400.0,
        "river_water_level_m": 2.0,
        "soil_moisture": 0.25,
        "elevation_m": 430.0,
        "historical_flood_events": 1
      }
    ]
  }'
```

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /model_info` - Model information
- `POST /predict` - Single prediction
- `POST /batch_predict` - Batch predictions

## Module Details

### flood_dataset.py

Generates synthetic flood data for India with realistic parameters:
- **18 districts** across major river basins
- **Features**: rainfall, river level, soil moisture, elevation, historical floods
- **Target**: binary flood occurrence
- **Geographic realism**: Different risk levels by basin and region

### flood_preprocessing.py

Data preprocessing pipeline:
- **Label encoding** for categorical features (district, state, basin)
- **Standard scaling** for numeric features
- **Feature engineering**: rainfall categories, river danger zones, soil saturation
- **Train/test split** with stratification

### flood_train.py

XGBoost training pipeline:
- **GridSearchCV** for hyperparameter tuning
- **Cross-validation** (5-fold)
- **Feature importance** extraction
- **Model persistence** (JSON format)
- **Training history** logging

### flood_evaluate.py

Comprehensive model evaluation:
- **Metrics**: accuracy, precision, recall, F1, ROC-AUC
- **Visualizations**: confusion matrix, ROC curve, PR curve
- **Risk categorization**: Low, Moderate, High, Very High
- **Report generation**: text and plots

### flood_api.py

FastAPI REST endpoint:
- **Single/batch predictions**
- **Input validation** with Pydantic
- **CORS support**
- **Health checks**
- **Model info endpoint**

## Risk Categories

- **Low**: < 25% probability
- **Moderate**: 25-50% probability
- **High**: 50-75% probability
- **Very High**: > 75% probability

## Model Performance

Expected performance on synthetic data:
- Accuracy: ~85-90%
- F1 Score: ~0.80-0.85
- ROC-AUC: ~0.90-0.95

## File Structure

```
ml_model/
├── flood_dataset.py          # Dataset generator
├── flood_preprocessing.py    # Preprocessing pipeline
├── flood_train.py           # Training pipeline
├── flood_evaluate.py        # Evaluation module
├── flood_api.py             # FastAPI endpoint
├── flood_requirements.txt    # Dependencies
├── FLOOD_README.md          # This file
└── saved_models/            # Model artifacts
    ├── flood_dataset.csv
    ├── flood_xgboost_model.json
    ├── flood_xgboost_model_history.json
    ├── flood_preprocessor.pkl
    └── evaluation/
        ├── confusion_matrix.png
        ├── roc_curve.png
        └── precision_recall_curve.png
```

## Customization

### Add New Districts

Edit `FLOOD_REGIONS` in `flood_dataset.py`:

```python
{"name": "YourDistrict", "state": "YourState", "basin": "BasinName", 
 "rainfall_mean": 1000, "rainfall_std": 200, "river_level_mean": 10.0, 
 "river_level_std": 3.0, "soil_moisture_mean": 0.55, "elevation_mean": 100, 
 "historical_flood_rate": 0.15, "base_risk": 0.20}
```

### Adjust Hyperparameters

Modify `param_grid` in `flood_train.py`:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    # ... add more parameters
}
```

### Change Risk Thresholds

Modify `get_risk_category()` in `flood_api.py`:

```python
if probability < 30:  # Adjust thresholds
    return "Low"
elif probability < 60:
    return "Moderate"
# ...
```

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY ml_model/ .
RUN pip install -r flood_requirements.txt
CMD ["python", "flood_api.py"]
```

### Using Gunicorn

```bash
gunicorn flood_api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

## Notes

- The dataset is synthetic for demonstration purposes
- For production use, replace with real historical flood data
- Model performance depends on data quality and feature engineering
- Consider adding temporal features (seasonality, trends) for real-world deployment
- Monitor model performance and retrain periodically with new data

## License

This is part of the Crisis AI Response project.
