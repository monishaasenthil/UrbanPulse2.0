"""
Unified Data Fetcher - Combines NYC Collisions and Weather APIs
Creates the multimodal data stream for Urban Pulse
"""
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_acquisition.nyc_collisions_api import NYCCollisionsAPI
from data_acquisition.weather_api import OpenMeteoAPI
from config.settings import BRONZE_DIR, NYC_LAT, NYC_LON


class DataFetcher:
    """
    Unified data fetcher that combines traffic and weather data
    into a multimodal dataset for Urban Pulse
    """
    
    def __init__(self):
        self.collisions_api = NYCCollisionsAPI()
        self.weather_api = OpenMeteoAPI()
        
    def fetch_unified_dataset(self, days_back=7, collision_limit=5000):
        """
        Fetch and merge collision and weather data
        
        Args:
            days_back: Number of days of historical data
            collision_limit: Maximum collision records
            
        Returns:
            Tuple of (collisions_df, weather_df, merged_df)
        """
        print("=" * 60)
        print("URBAN PULSE - DATA ACQUISITION")
        print("=" * 60)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Fetch collision data
        print(f"\n[1/3] Fetching collision data ({days_back} days)...")
        collisions_df = self.collisions_api.fetch_by_date_range(
            start_date, end_date, limit=collision_limit
        )
        print(f"      → Retrieved {len(collisions_df)} collision records")
        
        # Fetch weather data
        print(f"\n[2/3] Fetching weather data...")
        weather_df = self.weather_api.fetch_historical_weather(
            latitude=NYC_LAT,
            longitude=NYC_LON,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        print(f"      → Retrieved {len(weather_df)} weather records")
        
        # Merge datasets
        print(f"\n[3/3] Merging datasets...")
        merged_df = self._merge_datasets(collisions_df, weather_df)
        print(f"      → Created unified dataset with {len(merged_df)} records")
        
        print("\n" + "=" * 60)
        print("DATA ACQUISITION COMPLETE")
        print("=" * 60)
        
        return collisions_df, weather_df, merged_df
    
    def fetch_realtime_update(self):
        """
        Fetch latest data for real-time updates
        
        Returns:
            Tuple of (collisions_df, weather_dict)
        """
        # Get recent collisions (last 24 hours)
        collisions_df = self.collisions_api.fetch_recent_collisions(limit=100)
        
        # Get current weather
        weather_dict = self.weather_api.fetch_current_weather()
        
        return collisions_df, weather_dict
    
    def _merge_datasets(self, collisions_df, weather_df):
        """
        Merge collision and weather data by timestamp
        
        Args:
            collisions_df: Collision DataFrame
            weather_df: Weather DataFrame
            
        Returns:
            Merged DataFrame
        """
        if collisions_df.empty or weather_df.empty:
            return collisions_df
            
        # Ensure datetime columns
        collisions_df = collisions_df.copy()
        weather_df = weather_df.copy()
        
        # Round collision datetime to nearest hour for merging
        # Use 'h' instead of 'H' for pandas 2.0+ compatibility
        collisions_df['merge_hour'] = collisions_df['crash_datetime'].dt.floor('h')
        weather_df['merge_hour'] = weather_df['timestamp'].dt.floor('h')
        
        # Select weather columns for merge
        weather_cols = ['merge_hour', 'temperature', 'humidity', 'precipitation', 
                       'rain', 'wind_speed', 'visibility', 'is_raining', 'is_heavy_rain']
        weather_for_merge = weather_df[[c for c in weather_cols if c in weather_df.columns]]
        
        # Merge on hour
        merged_df = collisions_df.merge(
            weather_for_merge,
            on='merge_hour',
            how='left'
        )
        
        # Fill missing weather data with defaults
        weather_defaults = {
            'temperature': 20.0,
            'humidity': 50.0,
            'precipitation': 0.0,
            'rain': 0.0,
            'wind_speed': 10.0,
            'visibility': 10000.0,
            'is_raining': False,
            'is_heavy_rain': False
        }
        
        for col, default in weather_defaults.items():
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(default)
        
        return merged_df
    
    def save_bronze_data(self, collisions_df, weather_df):
        """
        Save raw data to bronze layer
        
        Args:
            collisions_df: Raw collision data
            weather_df: Raw weather data
        """
        os.makedirs(BRONZE_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save collisions
        collisions_path = os.path.join(BRONZE_DIR, f"collisions_bronze_{timestamp}.csv")
        collisions_df.to_csv(collisions_path, index=False)
        print(f"Saved: {collisions_path}")
        
        # Save weather
        weather_path = os.path.join(BRONZE_DIR, f"weather_bronze_{timestamp}.csv")
        weather_df.to_csv(weather_path, index=False)
        print(f"Saved: {weather_path}")
        
        return collisions_path, weather_path


def main():
    """Main function to test data fetching"""
    fetcher = DataFetcher()
    
    # Fetch unified dataset
    collisions_df, weather_df, merged_df = fetcher.fetch_unified_dataset(days_back=7)
    
    # Save to bronze layer
    if not collisions_df.empty:
        fetcher.save_bronze_data(collisions_df, weather_df)
    
    # Display sample
    print("\n" + "=" * 60)
    print("SAMPLE DATA")
    print("=" * 60)
    
    if not merged_df.empty:
        print("\nMerged Dataset Columns:")
        print(merged_df.columns.tolist())
        print(f"\nSample Records:")
        print(merged_df.head())
    
    return merged_df


if __name__ == "__main__":
    main()
