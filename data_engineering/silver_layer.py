"""
Silver Layer - Cleaned and Transformed Data
Applies data cleaning, type conversion, deduplication, and timestamp alignment
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SILVER_DIR
from data_engineering.h3_processor import H3Processor


class SilverLayer:
    """
    Silver Layer for cleaned and transformed data
    - Data cleaning and validation
    - Type conversion
    - Deduplication
    - Timestamp alignment
    - H3 micro-zoning
    """
    
    def __init__(self):
        self.silver_dir = SILVER_DIR
        os.makedirs(self.silver_dir, exist_ok=True)
        self.h3_processor = H3Processor()
        
    def process_collisions(self, df):
        """
        Clean and transform collision data
        
        Args:
            df: Raw collision DataFrame from bronze layer
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
            
        df_clean = df.copy()
        
        # Step 1: Remove duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        print(f"  Deduplication: {initial_count} → {len(df_clean)} records")
        
        # Step 2: Clean coordinates
        df_clean['latitude'] = pd.to_numeric(df_clean['latitude'], errors='coerce')
        df_clean['longitude'] = pd.to_numeric(df_clean['longitude'], errors='coerce')
        
        # Remove invalid coordinates
        df_clean = df_clean[
            (df_clean['latitude'].notna()) & 
            (df_clean['longitude'].notna()) &
            (df_clean['latitude'] != 0) &
            (df_clean['longitude'] != 0)
        ]
        print(f"  Coordinate cleaning: {len(df_clean)} valid records")
        
        # Step 3: Parse and validate datetime
        if 'crash_date' in df_clean.columns:
            df_clean['crash_date'] = pd.to_datetime(df_clean['crash_date'], errors='coerce')
        
        if 'crash_datetime' in df_clean.columns:
            df_clean['crash_datetime'] = pd.to_datetime(df_clean['crash_datetime'], errors='coerce')
        elif 'crash_date' in df_clean.columns and 'crash_time' in df_clean.columns:
            df_clean['crash_datetime'] = df_clean.apply(
                lambda row: self._combine_datetime(row['crash_date'], row['crash_time']),
                axis=1
            )
        
        # Step 4: Clean numeric columns
        numeric_cols = ['persons_injured', 'persons_killed', 'severity']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
        
        # Step 5: Clean categorical columns
        if 'borough' in df_clean.columns:
            df_clean['borough'] = df_clean['borough'].fillna('UNKNOWN').str.upper().str.strip()
        
        if 'contributing_factor' in df_clean.columns:
            df_clean['contributing_factor'] = df_clean['contributing_factor'].fillna('Unspecified')
        
        if 'vehicle_type' in df_clean.columns:
            df_clean['vehicle_type'] = df_clean['vehicle_type'].fillna('Unknown')
        
        # Step 6: Add H3 index
        df_clean = self.h3_processor.add_h3_column(df_clean)
        valid_h3 = df_clean['h3_index'].notna().sum()
        print(f"  H3 indexing: {valid_h3} records with valid H3")
        
        # Step 7: Add derived features
        if 'crash_datetime' in df_clean.columns:
            df_clean['hour'] = df_clean['crash_datetime'].dt.hour
            df_clean['day_of_week'] = df_clean['crash_datetime'].dt.dayofweek
            df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6])
            df_clean['is_peak_hour'] = df_clean['hour'].isin([7, 8, 9, 16, 17, 18, 19])
        
        # Step 8: Add processing metadata
        df_clean['_silver_timestamp'] = datetime.now()
        
        return df_clean
    
    def process_weather(self, df):
        """
        Clean and transform weather data
        
        Args:
            df: Raw weather DataFrame from bronze layer
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
            
        df_clean = df.copy()
        
        # Step 1: Remove duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        print(f"  Deduplication: {initial_count} → {len(df_clean)} records")
        
        # Step 2: Parse timestamp
        if 'timestamp' in df_clean.columns:
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce')
        
        # Step 3: Clean numeric columns
        weather_numeric = ['temperature', 'humidity', 'precipitation', 'rain', 
                          'wind_speed', 'visibility', 'wind_direction']
        for col in weather_numeric:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Step 4: Fill missing values with sensible defaults
        defaults = {
            'temperature': 20.0,
            'humidity': 50.0,
            'precipitation': 0.0,
            'rain': 0.0,
            'wind_speed': 10.0,
            'visibility': 10000.0
        }
        for col, default in defaults.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(default)
        
        # Step 5: Add derived weather features
        if 'precipitation' in df_clean.columns:
            df_clean['is_raining'] = df_clean['precipitation'] > 0
            df_clean['is_heavy_rain'] = df_clean['precipitation'] > 5.0
            df_clean['rain_intensity'] = pd.cut(
                df_clean['precipitation'],
                bins=[-np.inf, 0, 0.5, 2.5, 7.5, np.inf],
                labels=['none', 'light', 'moderate', 'heavy', 'extreme']
            )
        
        if 'wind_speed' in df_clean.columns:
            df_clean['is_windy'] = df_clean['wind_speed'] > 30
            df_clean['wind_category'] = pd.cut(
                df_clean['wind_speed'],
                bins=[-np.inf, 10, 20, 40, 60, np.inf],
                labels=['calm', 'light', 'moderate', 'strong', 'severe']
            )
        
        if 'visibility' in df_clean.columns:
            df_clean['low_visibility'] = df_clean['visibility'] < 1000
        
        # Step 6: Add temporal features
        if 'timestamp' in df_clean.columns:
            df_clean['hour'] = df_clean['timestamp'].dt.hour
            df_clean['date'] = df_clean['timestamp'].dt.date
        
        # Step 7: Add processing metadata
        df_clean['_silver_timestamp'] = datetime.now()
        
        return df_clean
    
    def merge_collision_weather(self, collision_df, weather_df):
        """
        Merge collision and weather data by timestamp
        
        Args:
            collision_df: Cleaned collision DataFrame
            weather_df: Cleaned weather DataFrame
            
        Returns:
            Merged DataFrame
        """
        if collision_df.empty or weather_df.empty:
            return collision_df
        
        # Round to nearest hour for merging
        collision_df = collision_df.copy()
        weather_df = weather_df.copy()
        
        collision_df['merge_hour'] = collision_df['crash_datetime'].dt.floor('h')
        weather_df['merge_hour'] = weather_df['timestamp'].dt.floor('h')
        
        # Select weather columns for merge
        weather_cols = ['merge_hour', 'temperature', 'humidity', 'precipitation', 
                       'rain', 'wind_speed', 'visibility', 'is_raining', 
                       'is_heavy_rain', 'rain_intensity', 'is_windy']
        weather_subset = weather_df[[c for c in weather_cols if c in weather_df.columns]]
        weather_subset = weather_subset.drop_duplicates(subset=['merge_hour'])
        
        # Merge
        merged = collision_df.merge(weather_subset, on='merge_hour', how='left')
        
        # Fill missing weather with defaults
        weather_defaults = {
            'temperature': 20.0,
            'humidity': 50.0,
            'precipitation': 0.0,
            'rain': 0.0,
            'wind_speed': 10.0,
            'visibility': 10000.0,
            'is_raining': False,
            'is_heavy_rain': False,
            'is_windy': False
        }
        for col, default in weather_defaults.items():
            if col in merged.columns:
                merged[col] = merged[col].fillna(default)
        
        return merged
    
    def save(self, df, name):
        """
        Save processed data to silver layer
        
        Args:
            df: Processed DataFrame
            name: Dataset name (e.g., 'collisions', 'weather', 'merged')
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_silver_{timestamp}.csv"
        filepath = os.path.join(self.silver_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"Silver Layer: Saved {len(df)} records to {filename}")
        
        return filepath
    
    def load_latest(self, name):
        """Load most recent silver data by name"""
        prefix = f"{name}_silver"
        files = [f for f in os.listdir(self.silver_dir) 
                if f.startswith(prefix) and f.endswith('.csv')]
        
        if not files:
            return pd.DataFrame()
        
        latest = max(files)
        return pd.read_csv(os.path.join(self.silver_dir, latest))
    
    def _combine_datetime(self, date, time_str):
        """Combine date and time into datetime"""
        try:
            if pd.isna(date):
                return None
            if pd.isna(time_str) or not time_str:
                return date
            time_parts = str(time_str).split(':')
            hour = int(time_parts[0])
            minute = int(time_parts[1]) if len(time_parts) > 1 else 0
            return date.replace(hour=hour, minute=minute)
        except:
            return date
