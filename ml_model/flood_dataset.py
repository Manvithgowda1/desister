"""
flood_dataset.py - Synthetic Flood Dataset Generator for India
Generates realistic flood risk samples across India's major river basins and districts.
"""

import numpy as np
import pandas as pd
import os

# Reproducibility
SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------------
# India Flood-Prone Regions and Districts
# ------------------------------------------------------------------
# Major river basins and flood-prone districts with realistic parameters
FLOOD_REGIONS = [
    # Ganga Basin - High Flood Risk
    {"name": "Patna", "state": "Bihar", "basin": "Ganga", "rainfall_mean": 1200, "rainfall_std": 300, 
     "river_level_mean": 15.0, "river_level_std": 5.0, "soil_moisture_mean": 0.65, "elevation_mean": 55, 
     "historical_flood_rate": 0.35, "base_risk": 0.40},
    {"name": "Varanasi", "state": "Uttar Pradesh", "basin": "Ganga", "rainfall_mean": 1050, "rainfall_std": 250,
     "river_level_mean": 12.0, "river_level_std": 4.0, "soil_moisture_mean": 0.60, "elevation_mean": 80,
     "historical_flood_rate": 0.25, "base_risk": 0.30},
    {"name": "Allahabad", "state": "Uttar Pradesh", "basin": "Ganga", "rainfall_mean": 980, "rainfall_std": 220,
     "river_level_mean": 11.0, "river_level_std": 3.5, "soil_moisture_mean": 0.58, "elevation_mean": 90,
     "historical_flood_rate": 0.20, "base_risk": 0.25},
    
    # Brahmaputra Basin - Very High Flood Risk
    {"name": "Dibrugarh", "state": "Assam", "basin": "Brahmaputra", "rainfall_mean": 2800, "rainfall_std": 500,
     "river_level_mean": 18.0, "river_level_std": 6.0, "soil_moisture_mean": 0.75, "elevation_mean": 95,
     "historical_flood_rate": 0.50, "base_risk": 0.55},
    {"name": "Guwahati", "state": "Assam", "basin": "Brahmaputra", "rainfall_mean": 1700, "rainfall_std": 400,
     "river_level_mean": 14.0, "river_level_std": 5.0, "soil_moisture_mean": 0.70, "elevation_mean": 55,
     "historical_flood_rate": 0.40, "base_risk": 0.45},
    {"name": "Majuli", "state": "Assam", "basin": "Brahmaputra", "rainfall_mean": 2200, "rainfall_std": 450,
     "river_level_mean": 16.0, "river_level_std": 5.5, "soil_moisture_mean": 0.72, "elevation_mean": 85,
     "historical_flood_rate": 0.55, "base_risk": 0.60},
    
    # Godavari Basin - Moderate to High Flood Risk
    {"name": "Rajahmundry", "state": "Andhra Pradesh", "basin": "Godavari", "rainfall_mean": 1100, "rainfall_std": 280,
     "river_level_mean": 10.0, "river_level_std": 3.5, "soil_moisture_mean": 0.55, "elevation_mean": 15,
     "historical_flood_rate": 0.20, "base_risk": 0.25},
    {"name": "Nashik", "state": "Maharashtra", "basin": "Godavari", "rainfall_mean": 900, "rainfall_std": 200,
     "river_level_mean": 8.0, "river_level_std": 2.5, "soil_moisture_mean": 0.50, "elevation_mean": 580,
     "historical_flood_rate": 0.12, "base_risk": 0.15},
    
    # Krishna Basin - Moderate Flood Risk
    {"name": "Vijayawada", "state": "Andhra Pradesh", "basin": "Krishna", "rainfall_mean": 950, "rainfall_std": 230,
     "river_level_mean": 9.0, "river_level_std": 3.0, "soil_moisture_mean": 0.52, "elevation_mean": 12,
     "historical_flood_rate": 0.18, "base_risk": 0.20},
    {"name": "Sangli", "state": "Maharashtra", "basin": "Krishna", "rainfall_mean": 800, "rainfall_std": 180,
     "river_level_mean": 7.0, "river_level_std": 2.0, "soil_moisture_mean": 0.48, "elevation_mean": 550,
     "historical_flood_rate": 0.10, "base_risk": 0.12},
    
    # Mahanadi Basin - Moderate Flood Risk
    {"name": "Cuttack", "state": "Odisha", "basin": "Mahanadi", "rainfall_mean": 1450, "rainfall_std": 320,
     "river_level_mean": 11.0, "river_level_std": 4.0, "soil_moisture_mean": 0.60, "elevation_mean": 25,
     "historical_flood_rate": 0.25, "base_risk": 0.28},
    {"name": "Sambalpur", "state": "Odisha", "basin": "Mahanadi", "rainfall_mean": 1350, "rainfall_std": 300,
     "river_level_mean": 10.0, "river_level_std": 3.5, "soil_moisture_mean": 0.58, "elevation_mean": 150,
     "historical_flood_rate": 0.20, "base_risk": 0.22},
    
    # Kaveri Basin - Low to Moderate Flood Risk
    {"name": "Trichy", "state": "Tamil Nadu", "basin": "Kaveri", "rainfall_mean": 850, "rainfall_std": 200,
     "river_level_mean": 7.0, "river_level_std": 2.5, "soil_moisture_mean": 0.45, "elevation_mean": 78,
     "historical_flood_rate": 0.12, "base_risk": 0.15},
    {"name": "Mysore", "state": "Karnataka", "basin": "Kaveri", "rainfall_mean": 750, "rainfall_std": 180,
     "river_level_mean": 6.0, "river_level_std": 2.0, "soil_moisture_mean": 0.42, "elevation_mean": 770,
     "historical_flood_rate": 0.08, "base_risk": 0.10},
    
    # Coastal Areas - Moderate Flood Risk (Cyclone-induced)
    {"name": "Kolkata", "state": "West Bengal", "basin": "Coastal", "rainfall_mean": 1600, "rainfall_std": 350,
     "river_level_mean": 8.0, "river_level_std": 3.0, "soil_moisture_mean": 0.62, "elevation_mean": 8,
     "historical_flood_rate": 0.22, "base_risk": 0.25},
    {"name": "Chennai", "state": "Tamil Nadu", "basin": "Coastal", "rainfall_mean": 1300, "rainfall_std": 300,
     "river_level_mean": 5.0, "river_level_std": 1.5, "soil_moisture_mean": 0.50, "elevation_mean": 6,
     "historical_flood_rate": 0.15, "base_risk": 0.18},
    {"name": "Mumbai", "state": "Maharashtra", "basin": "Coastal", "rainfall_mean": 2200, "rainfall_std": 500,
     "river_level_mean": 4.0, "river_level_std": 1.0, "soil_moisture_mean": 0.55, "elevation_mean": 14,
     "historical_flood_rate": 0.18, "base_risk": 0.20},
    
    # Rajasthan - Low Flood Risk (Arid region)
    {"name": "Jaipur", "state": "Rajasthan", "basin": "Arid", "rainfall_mean": 600, "rainfall_std": 150,
     "river_level_mean": 3.0, "river_level_std": 1.0, "soil_moisture_mean": 0.30, "elevation_mean": 430,
     "historical_flood_rate": 0.05, "base_risk": 0.06},
    {"name": "Jodhpur", "state": "Rajasthan", "basin": "Arid", "rainfall_mean": 350, "rainfall_std": 100,
     "river_level_mean": 2.0, "river_level_std": 0.8, "soil_moisture_mean": 0.25, "elevation_mean": 230,
     "historical_flood_rate": 0.03, "base_risk": 0.04},
]


def generate_dataset(n_samples=5000):
    """
    Generate a synthetic flood dataset with realistic Indian region parameters.
    
    Returns:
        pd.DataFrame with columns: district, state, basin, rainfall_mm, 
        river_water_level_m, soil_moisture, elevation_m, historical_flood_events,
        flood_occurred
    """
    records = []
    samples_per_region = n_samples // len(FLOOD_REGIONS)
    remainder = n_samples % len(FLOOD_REGIONS)
    
    for i, region in enumerate(FLOOD_REGIONS):
        # Distribute remainder samples across first few regions
        n = samples_per_region + (1 if i < remainder else 0)
        
        # Rainfall (Normal distribution with region-specific parameters)
        rainfall = np.clip(
            np.random.normal(region["rainfall_mean"], region["rainfall_std"], n),
            100.0, 4000.0
        )
        
        # River water level (Normal distribution)
        river_level = np.clip(
            np.random.normal(region["river_level_mean"], region["river_level_std"], n),
            0.5, 30.0
        )
        
        # Soil moisture (Beta distribution scaled to 0-1)
        soil_moisture = np.clip(
            np.random.beta(region["soil_moisture_mean"] * 2, (1 - region["soil_moisture_mean"]) * 2, n),
            0.1, 0.95
        )
        
        # Elevation (Normal distribution with slight variation)
        elevation = np.clip(
            np.random.normal(region["elevation_mean"], region["elevation_mean"] * 0.1, n),
            0.0, 2000.0
        )
        
        # Historical flood events (Poisson distribution)
        historical_floods = np.random.poisson(lam=region["historical_flood_rate"] * 10, size=n)
        
        # --- Target: flood_occurred ---
        # Base probability depends on region's risk level
        base_p = region["base_risk"]
        
        # Modifiers based on features
        rainfall_modifier = np.clip((rainfall - 500) / 3000 * 0.3, 0.0, 0.3)
        river_modifier = np.clip((river_level - 5) / 25 * 0.25, 0.0, 0.25)
        soil_modifier = np.clip((soil_moisture - 0.3) / 0.7 * 0.15, 0.0, 0.15)
        elevation_modifier = np.clip((500 - elevation) / 500 * 0.1, 0.0, 0.1)
        historical_modifier = np.clip(historical_floods / 20 * 0.1, 0.0, 0.1)
        
        prob = np.clip(
            base_p + rainfall_modifier + river_modifier + soil_modifier + 
            elevation_modifier + historical_modifier, 
            0.01, 0.95
        )
        flood_occurred = (np.random.random(n) < prob).astype(int)
        
        for j in range(n):
            records.append({
                "district": region["name"],
                "state": region["state"],
                "basin": region["basin"],
                "rainfall_mm": round(rainfall[j], 2),
                "river_water_level_m": round(river_level[j], 2),
                "soil_moisture": round(soil_moisture[j], 3),
                "elevation_m": round(elevation[j], 1),
                "historical_flood_events": int(historical_floods[j]),
                "flood_occurred": int(flood_occurred[j])
            })
    
    df = pd.DataFrame(records)
    
    # Shuffle the dataset
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    
    return df


def save_dataset(df, output_dir=None):
    """Save dataset to CSV in the saved_models directory."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    os.makedirs(output_dir, exist_ok=True)
    
    path = os.path.join(output_dir, "flood_dataset.csv")
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path}")
    return path


if __name__ == "__main__":
    print("Generating synthetic flood dataset for India...")
    df = generate_dataset(n_samples=5000)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df["flood_occurred"].value_counts())
    print(f"\nPositive class ratio: {df['flood_occurred'].mean():.2%}")
    print(f"\nState distribution:")
    print(df["state"].value_counts())
    print(f"\nBasin distribution:")
    print(df["basin"].value_counts())
    print(f"\nSample rows:")
    print(df.head(10).to_string())
    
    save_dataset(df)
