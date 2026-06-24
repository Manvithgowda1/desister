"""
dataset.py - Synthetic Earthquake Dataset Generator for India
Generates geographically realistic samples across India's five BIS seismic zones.
"""

import numpy as np
import pandas as pd
import os

# Reproducibility
SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------------
# India Seismic Zone Mapping (BIS IS:1893)
# ------------------------------------------------------------------
# Each entry: (lat_min, lat_max, lon_min, lon_max, zone, base_eq_rate, avg_mag, fault_dist_mean)
SEISMIC_REGIONS = [
    # Zone V - Very High Risk (Northeast India, J&K, Himachal, Uttarakhand, Kutch)
    {"name": "Northeast India",      "lat": (25.0, 28.5), "lon": (89.0, 97.0), "zone": 5, "eq_rate": 8.0,  "mag_mean": 5.8, "fault_km": 10,  "pop_density": 400},
    {"name": "Kashmir",              "lat": (33.0, 37.0), "lon": (73.0, 78.0), "zone": 5, "eq_rate": 7.0,  "mag_mean": 5.5, "fault_km": 15,  "pop_density": 150},
    {"name": "Uttarakhand",          "lat": (29.0, 31.5), "lon": (77.5, 81.0), "zone": 5, "eq_rate": 6.5,  "mag_mean": 5.3, "fault_km": 12,  "pop_density": 200},
    {"name": "Kutch Gujarat",        "lat": (22.5, 24.5), "lon": (68.5, 72.0), "zone": 5, "eq_rate": 5.0,  "mag_mean": 5.6, "fault_km": 20,  "pop_density": 350},
    
    # Zone IV - High Risk (Delhi-NCR, Bihar plains, parts of Maharashtra, Himachal foothills)
    {"name": "Delhi NCR",            "lat": (28.0, 29.5), "lon": (76.5, 78.0), "zone": 4, "eq_rate": 4.0,  "mag_mean": 4.5, "fault_km": 30,  "pop_density": 11320},
    {"name": "Bihar Plains",         "lat": (24.5, 27.5), "lon": (83.0, 88.5), "zone": 4, "eq_rate": 3.5,  "mag_mean": 4.2, "fault_km": 35,  "pop_density": 1100},
    {"name": "Himachal Foothills",   "lat": (30.5, 33.0), "lon": (75.5, 78.5), "zone": 4, "eq_rate": 4.5,  "mag_mean": 4.8, "fault_km": 25,  "pop_density": 120},
    {"name": "Maharashtra West",     "lat": (16.5, 20.0), "lon": (72.5, 75.0), "zone": 4, "eq_rate": 2.5,  "mag_mean": 4.0, "fault_km": 50,  "pop_density": 600},
    
    # Zone III - Moderate Risk (Most of peninsular India, Kolkata, parts of Rajasthan)
    {"name": "Karnataka",            "lat": (12.0, 16.5), "lon": (74.0, 78.5), "zone": 3, "eq_rate": 1.5,  "mag_mean": 3.5, "fault_km": 80,  "pop_density": 320},
    {"name": "Tamil Nadu Coast",     "lat": (8.5,  13.5), "lon": (78.0, 80.5), "zone": 3, "eq_rate": 1.8,  "mag_mean": 3.8, "fault_km": 70,  "pop_density": 550},
    {"name": "Kolkata Region",       "lat": (21.5, 24.0), "lon": (86.5, 89.0), "zone": 3, "eq_rate": 2.0,  "mag_mean": 3.6, "fault_km": 60,  "pop_density": 7500},
    {"name": "Rajasthan",            "lat": (24.5, 30.0), "lon": (69.5, 76.5), "zone": 3, "eq_rate": 1.2,  "mag_mean": 3.3, "fault_km": 90,  "pop_density": 200},
    
    # Zone II - Low Risk (Most of central-south India, stable continental shield)
    {"name": "Central India",        "lat": (20.0, 25.0), "lon": (75.0, 83.0), "zone": 2, "eq_rate": 0.5,  "mag_mean": 2.8, "fault_km": 150, "pop_density": 300},
    {"name": "Kerala",               "lat": (8.0,  12.5), "lon": (75.0, 77.5), "zone": 2, "eq_rate": 0.6,  "mag_mean": 2.5, "fault_km": 120, "pop_density": 860},
    {"name": "Andhra Pradesh",       "lat": (13.5, 19.5), "lon": (77.0, 84.0), "zone": 2, "eq_rate": 0.4,  "mag_mean": 2.6, "fault_km": 130, "pop_density": 310},
]


def generate_dataset(n_samples=5000):
    """
    Generate a synthetic earthquake dataset with realistic Indian seismic zone parameters.
    
    Returns:
        pd.DataFrame with columns: latitude, longitude, seismic_zone, 
        historical_eq_frequency, avg_historical_magnitude, fault_proximity_km,
        population_density, earthquake_occurred
    """
    records = []
    samples_per_region = n_samples // len(SEISMIC_REGIONS)
    remainder = n_samples % len(SEISMIC_REGIONS)
    
    for i, region in enumerate(SEISMIC_REGIONS):
        # Distribute remainder samples across first few regions
        n = samples_per_region + (1 if i < remainder else 0)
        
        # Sample coordinates within region bounds
        lats = np.random.uniform(region["lat"][0], region["lat"][1], n)
        lons = np.random.uniform(region["lon"][0], region["lon"][1], n)
        
        # Seismic zone (fixed per region with minor jitter for border areas)
        zones = np.full(n, region["zone"])
        
        # Historical earthquake frequency (Poisson distribution centered on region's rate)
        eq_freq = np.random.poisson(lam=region["eq_rate"], size=n)
        
        # Average historical magnitude (Normal distribution with zone-appropriate parameters)
        mag_std = 0.6 if region["zone"] >= 4 else 0.4
        avg_mag = np.clip(np.random.normal(region["mag_mean"], mag_std, n), 1.5, 8.5)
        
        # Fault proximity (Exponential distribution — closer faults are more dangerous)
        fault_km = np.clip(np.random.exponential(scale=region["fault_km"], size=n), 1.0, 500.0)
        
        # Population density (Log-normal around region's mean)
        pop_density = np.clip(
            np.random.lognormal(mean=np.log(region["pop_density"]), sigma=0.5, size=n),
            10.0, 50000.0
        )
        
        # --- Target: earthquake_occurred ---
        # Base probability depends on seismic zone, modified by frequency and fault proximity
        zone_base_prob = {5: 0.45, 4: 0.30, 3: 0.15, 2: 0.05}
        base_p = zone_base_prob[region["zone"]]
        
        # Modifiers: higher freq -> higher prob, closer faults -> higher prob
        freq_modifier = np.clip(eq_freq / 10.0, 0.0, 0.3)
        fault_modifier = np.clip(0.2 - (fault_km / 500.0) * 0.2, 0.0, 0.2)
        mag_modifier = np.clip((avg_mag - 3.0) / 15.0, 0.0, 0.15)
        
        prob = np.clip(base_p + freq_modifier + fault_modifier + mag_modifier, 0.01, 0.95)
        eq_occurred = (np.random.random(n) < prob).astype(int)
        
        for j in range(n):
            records.append({
                "latitude": round(lats[j], 4),
                "longitude": round(lons[j], 4),
                "seismic_zone": int(zones[j]),
                "historical_eq_frequency": int(eq_freq[j]),
                "avg_historical_magnitude": round(avg_mag[j], 2),
                "fault_proximity_km": round(fault_km[j], 2),
                "population_density": round(pop_density[j], 1),
                "earthquake_occurred": int(eq_occurred[j])
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
    
    path = os.path.join(output_dir, "earthquake_dataset.csv")
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path}")
    return path


if __name__ == "__main__":
    print("Generating synthetic earthquake dataset for India...")
    df = generate_dataset(n_samples=5000)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df["earthquake_occurred"].value_counts())
    print(f"\nPositive class ratio: {df['earthquake_occurred'].mean():.2%}")
    print(f"\nSeismic zone distribution:")
    print(df["seismic_zone"].value_counts().sort_index())
    print(f"\nSample rows:")
    print(df.head(10).to_string())
    
    save_dataset(df)
