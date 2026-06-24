-- PostgreSQL database schema for CRISIS-AI Disaster Risk Prediction & Analysis

CREATE TABLE IF NOT EXISTS disaster_records (
    id SERIAL PRIMARY KEY,
    state VARCHAR(100) NOT NULL,
    district VARCHAR(100) NOT NULL,
    latitude DECIMAL(9,6) NOT NULL,
    longitude DECIMAL(9,6) NOT NULL,
    date DATE NOT NULL,
    disaster_type VARCHAR(50),      -- 'earthquake', 'flood', 'cyclone', 'landslide', 'drought', 'heatwave', or NULL
    rainfall DECIMAL(7,2),          -- Daily rainfall in mm
    temperature DECIMAL(5,2),       -- Daily average temperature in °C
    humidity DECIMAL(5,2),          -- Relative humidity percentage
    magnitude DECIMAL(3,1),         -- Earthquake magnitude (Richter scale)
    disaster_occurred BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_daily_location UNIQUE (latitude, longitude, date)
);

-- Indexing for fast search / predictive modeling query patterns
CREATE INDEX IF NOT EXISTS idx_location_date ON disaster_records(latitude, longitude, date);
CREATE INDEX IF NOT EXISTS idx_state_district ON disaster_records(state, district);
CREATE INDEX IF NOT EXISTS idx_disaster ON disaster_records(disaster_type) WHERE disaster_occurred = TRUE;
