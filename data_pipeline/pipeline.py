import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime

try:
    import psycopg2
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

# Import configurations & extractors
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from extractors.earthquake_ncs import fetch_historical_earthquakes
from extractors.weather_imd import fetch_weather_grid, download_imdlib_historical_rain

def get_db_connection():
    """
    Connect to PostgreSQL database using config.
    Falls back to a local SQLite database if PostgreSQL connection fails.
    """
    if HAS_POSTGRES:
        try:
            conn = psycopg2.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                dbname=config.DB_NAME,
                user=config.DB_USER,
                password=config.DB_PASSWORD
            )
            print("Connected to PostgreSQL database successfully.")
            return conn, "postgresql"
        except Exception as e:
            print(f"Warning: PostgreSQL connection failed: {e}. Falling back to SQLite local database.")
    else:
        print("Warning: psycopg2 not installed. PostgreSQL mode disabled.")
    
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crisis_ai_local.db")
    conn = sqlite3.connect(db_path)
    print(f"Connected to SQLite database at {db_path}.")
    return conn, "sqlite"

def initialize_database(conn, db_type):
    """
    Creates target database schema tables and indexes.
    """
    cursor = conn.cursor()
    
    # Read schema.sql file
    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
        
    if db_type == "postgresql":
        try:
            cursor.execute(schema_sql)
            conn.commit()
            print("PostgreSQL tables and indices verified.")
        except Exception as e:
            conn.rollback()
            print(f"Error initializing PostgreSQL schema: {e}")
    else:
        # Translate PostgreSQL dialects to SQLite equivalents
        sqlite_schema = schema_sql.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
        sqlite_schema = sqlite_schema.replace("DECIMAL(9,6)", "REAL")
        sqlite_schema = sqlite_schema.replace("DECIMAL(7,2)", "REAL")
        sqlite_schema = sqlite_schema.replace("DECIMAL(5,2)", "REAL")
        sqlite_schema = sqlite_schema.replace("DECIMAL(3,1)", "REAL")
        sqlite_schema = sqlite_schema.replace("TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "DATETIME DEFAULT CURRENT_TIMESTAMP")
        
        # SQLite doesn't support ON CONFLICT index names directly in the same way, but supports table level unique constraints.
        # Run statements separated by semicolons
        try:
            for statement in sqlite_schema.split(";"):
                if statement.strip():
                    cursor.execute(statement)
            conn.commit()
            print("SQLite tables and indices verified.")
        except Exception as e:
            print(f"Error initializing SQLite schema: {e}")
            
    cursor.close()

def insert_records(conn, db_type, df):
    """
    Batch inserts/upserts records into the database.
    """
    if df.empty:
        print("No records to insert.")
        return
        
    cursor = conn.cursor()
    inserted_count = 0
    
    if db_type == "postgresql":
        insert_query = """
            INSERT INTO disaster_records 
            (state, district, latitude, longitude, date, disaster_type, rainfall, temperature, humidity, magnitude, disaster_occurred)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (latitude, longitude, date) 
            DO UPDATE SET 
                disaster_type = EXCLUDED.disaster_type,
                rainfall = COALESCE(EXCLUDED.rainfall, disaster_records.rainfall),
                temperature = COALESCE(EXCLUDED.temperature, disaster_records.temperature),
                humidity = COALESCE(EXCLUDED.humidity, disaster_records.humidity),
                magnitude = COALESCE(EXCLUDED.magnitude, disaster_records.magnitude),
                disaster_occurred = EXCLUDED.disaster_occurred;
        """
        for _, row in df.iterrows():
            try:
                cursor.execute(insert_query, (
                    row["state"], row["district"], row["latitude"], row["longitude"], row["date"],
                    row["disaster_type"], row["rainfall"], row["temperature"], row["humidity"],
                    row["magnitude"], row["disaster_occurred"]
                ))
                inserted_count += 1
            except Exception as e:
                print(f"Failed to insert row: {e}")
        conn.commit()
        
    else:
        # SQLite Upsert
        insert_query = """
            INSERT INTO disaster_records 
            (state, district, latitude, longitude, date, disaster_type, rainfall, temperature, humidity, magnitude, disaster_occurred)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(latitude, longitude, date) 
            DO UPDATE SET 
                disaster_type = excluded.disaster_type,
                rainfall = COALESCE(excluded.rainfall, disaster_records.rainfall),
                temperature = COALESCE(excluded.temperature, disaster_records.temperature),
                humidity = COALESCE(excluded.humidity, disaster_records.humidity),
                magnitude = COALESCE(excluded.magnitude, disaster_records.magnitude),
                disaster_occurred = excluded.disaster_occurred;
        """
        for _, row in df.iterrows():
            try:
                cursor.execute(insert_query, (
                    row["state"], row["district"], row["latitude"], row["longitude"], row["date"],
                    row["disaster_type"], row["rainfall"], row["temperature"], row["humidity"],
                    row["magnitude"], row["disaster_occurred"]
                ))
                inserted_count += 1
            except Exception as e:
                print(f"Failed to insert SQLite row: {e}")
        conn.commit()

    print(f"Upserted {inserted_count} rows into the database.")
    cursor.close()

def get_already_processed_records(conn):
    """
    Get a set of already processed (latitude, longitude, date) records from the database.
    """
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT latitude, longitude, date FROM disaster_records WHERE rainfall IS NOT NULL OR temperature IS NOT NULL")
        rows = cursor.fetchall()
    except Exception as e:
        print(f"Could not check processed records: {e}")
        rows = []
    cursor.close()
    
    processed = set()
    for lat, lon, date in rows:
        # handle date conversion safely
        if date:
            date_str = date if isinstance(date, str) else date.strftime('%Y-%m-%d')
            processed.add((round(float(lat), 4), round(float(lon), 4), date_str))
    return processed

def run_etl_pipeline():
    """
    Main orchestrator function for the ETL process with checkpoint resumption support.
    """
    print("\nStarting India Disaster Data Collection ETL Pipeline...")
    
    # 1. Establish Database Connection & Initialize Tables
    conn, db_type = get_db_connection()
    initialize_database(conn, db_type)
    
    # 2. Extract Seismological Records
    earthquake_df = fetch_historical_earthquakes()
    
    # 3. Filter already processed records (Resume checkpoint support)
    processed_set = get_already_processed_records(conn)
    print(f"Found {len(processed_set)} already enriched records in the database.")
    
    def is_unprocessed(row):
        key = (round(float(row["latitude"]), 4), round(float(row["longitude"]), 4), row["date"])
        return key not in processed_set
        
    if not earthquake_df.empty:
        unprocessed_df = earthquake_df[earthquake_df.apply(is_unprocessed, axis=1)]
    else:
        unprocessed_df = earthquake_df
        
    print(f"{len(unprocessed_df)} out of {len(earthquake_df)} records remain to be processed.")
    
    if unprocessed_df.empty:
        print("All records have been fully processed. Nothing to resume.")
        conn.close()
        return
        
    # 4. Enrich with Meteorological Data (Next Batch of 25)
    batch_size = 25
    records_to_process = unprocessed_df.head(batch_size)
    print(f"\nEnriching the next {len(records_to_process)} records from checkpoint...")
    
    enriched_records = []
    for idx, row in records_to_process.iterrows():
        weather = fetch_weather_grid(row["latitude"], row["longitude"], row["date"])
        
        row_dict = row.to_dict()
        row_dict["rainfall"] = weather["rainfall"]
        row_dict["temperature"] = weather["temperature"]
        row_dict["humidity"] = weather["humidity"]
        
        enriched_records.append(row_dict)
        
    enriched_df = pd.DataFrame(enriched_records)
    
    # 5. Ingest into Target Database
    insert_records(conn, db_type, enriched_df)
    
    # Close connection
    conn.close()
    print("ETL Pipeline execution completed successfully.\n")

if __name__ == "__main__":
    run_etl_pipeline()
