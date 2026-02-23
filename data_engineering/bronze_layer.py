"""
Bronze Layer - Raw Data Storage
Stores raw data from APIs with minimal transformation
"""
import pandas as pd
import os
import json
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import BRONZE_DIR


class BronzeLayer:
    """
    Bronze Layer for raw data storage
    - Stores data as-is from APIs
    - Adds metadata (fetch timestamp, source)
    - Maintains data lineage
    """
    
    def __init__(self):
        self.bronze_dir = BRONZE_DIR
        os.makedirs(self.bronze_dir, exist_ok=True)
        
    def save_collisions(self, df, source="nyc_api"):
        """
        Save raw collision data to bronze layer
        
        Args:
            df: Raw collision DataFrame
            source: Data source identifier
            
        Returns:
            Path to saved file
        """
        if df.empty:
            print("Warning: Empty DataFrame, skipping save")
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"collisions_bronze_{timestamp}.csv"
        filepath = os.path.join(self.bronze_dir, filename)
        
        # Add metadata
        df_save = df.copy()
        df_save['_bronze_timestamp'] = datetime.now()
        df_save['_source'] = source
        
        df_save.to_csv(filepath, index=False)
        
        # Save metadata
        self._save_metadata(filename, {
            'source': source,
            'record_count': len(df),
            'columns': list(df.columns),
            'timestamp': timestamp
        })
        
        print(f"Bronze Layer: Saved {len(df)} collision records to {filename}")
        return filepath
    
    def save_weather(self, df, source="open_meteo"):
        """
        Save raw weather data to bronze layer
        
        Args:
            df: Raw weather DataFrame
            source: Data source identifier
            
        Returns:
            Path to saved file
        """
        if df.empty:
            print("Warning: Empty DataFrame, skipping save")
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weather_bronze_{timestamp}.csv"
        filepath = os.path.join(self.bronze_dir, filename)
        
        # Add metadata
        df_save = df.copy()
        df_save['_bronze_timestamp'] = datetime.now()
        df_save['_source'] = source
        
        df_save.to_csv(filepath, index=False)
        
        # Save metadata
        self._save_metadata(filename, {
            'source': source,
            'record_count': len(df),
            'columns': list(df.columns),
            'timestamp': timestamp
        })
        
        print(f"Bronze Layer: Saved {len(df)} weather records to {filename}")
        return filepath
    
    def load_latest_collisions(self):
        """Load most recent collision data from bronze layer"""
        return self._load_latest("collisions_bronze")
    
    def load_latest_weather(self):
        """Load most recent weather data from bronze layer"""
        return self._load_latest("weather_bronze")
    
    def load_all_collisions(self):
        """Load and concatenate all collision data"""
        return self._load_all("collisions_bronze")
    
    def load_all_weather(self):
        """Load and concatenate all weather data"""
        return self._load_all("weather_bronze")
    
    def _load_latest(self, prefix):
        """Load most recent file with given prefix"""
        files = self._get_files_by_prefix(prefix)
        if not files:
            return pd.DataFrame()
        
        latest_file = max(files)
        filepath = os.path.join(self.bronze_dir, latest_file)
        return pd.read_csv(filepath)
    
    def _load_all(self, prefix):
        """Load and concatenate all files with given prefix"""
        files = self._get_files_by_prefix(prefix)
        if not files:
            return pd.DataFrame()
        
        dfs = []
        for f in files:
            filepath = os.path.join(self.bronze_dir, f)
            dfs.append(pd.read_csv(filepath))
        
        return pd.concat(dfs, ignore_index=True)
    
    def _get_files_by_prefix(self, prefix):
        """Get all files matching prefix"""
        if not os.path.exists(self.bronze_dir):
            return []
        return [f for f in os.listdir(self.bronze_dir) 
                if f.startswith(prefix) and f.endswith('.csv')]
    
    def _save_metadata(self, filename, metadata):
        """Save metadata for a bronze file"""
        metadata_dir = os.path.join(self.bronze_dir, "_metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        metadata_file = os.path.join(metadata_dir, f"{filename}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def get_data_summary(self):
        """Get summary of all bronze data"""
        collision_files = self._get_files_by_prefix("collisions_bronze")
        weather_files = self._get_files_by_prefix("weather_bronze")
        
        return {
            'collision_files': len(collision_files),
            'weather_files': len(weather_files),
            'latest_collision': max(collision_files) if collision_files else None,
            'latest_weather': max(weather_files) if weather_files else None
        }
