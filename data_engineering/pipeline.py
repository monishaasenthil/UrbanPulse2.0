"""
Data Pipeline - End-to-End Data Processing
Orchestrates the complete data flow from APIs to Gold layer
"""
import pandas as pd
import os
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_acquisition.data_fetcher import DataFetcher
from data_engineering.bronze_layer import BronzeLayer
from data_engineering.silver_layer import SilverLayer
from data_engineering.gold_layer import GoldLayer


class DataPipeline:
    """
    End-to-end data pipeline for Urban Pulse
    
    Flow:
    APIs → Bronze (Raw) → Silver (Clean) → Gold (Analytics-Ready)
    """
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.bronze = BronzeLayer()
        self.silver = SilverLayer()
        self.gold = GoldLayer()
        
    def run_full_pipeline(self, days_back=30, collision_limit=10000):
        """
        Run the complete data pipeline
        
        Args:
            days_back: Number of days of historical data
            collision_limit: Maximum collision records to fetch
            
        Returns:
            Dictionary with all processed DataFrames
        """
        print("=" * 70)
        print("URBAN PULSE - FULL DATA PIPELINE")
        print("=" * 70)
        start_time = datetime.now()
        
        results = {}
        
        # Phase 1: Data Acquisition
        print("\n" + "=" * 70)
        print("PHASE 1: DATA ACQUISITION")
        print("=" * 70)
        
        collisions_raw, weather_raw, _ = self.fetcher.fetch_unified_dataset(
            days_back=days_back,
            collision_limit=collision_limit
        )
        
        results['collisions_raw'] = collisions_raw
        results['weather_raw'] = weather_raw
        
        # Phase 2: Bronze Layer
        print("\n" + "=" * 70)
        print("PHASE 2: BRONZE LAYER (Raw Storage)")
        print("=" * 70)
        
        if not collisions_raw.empty:
            self.bronze.save_collisions(collisions_raw)
        if not weather_raw.empty:
            self.bronze.save_weather(weather_raw)
        
        # Phase 3: Silver Layer
        print("\n" + "=" * 70)
        print("PHASE 3: SILVER LAYER (Cleaning & Transformation)")
        print("=" * 70)
        
        print("\nProcessing collisions...")
        collisions_clean = self.silver.process_collisions(collisions_raw)
        
        print("\nProcessing weather...")
        weather_clean = self.silver.process_weather(weather_raw)
        
        print("\nMerging datasets...")
        merged = self.silver.merge_collision_weather(collisions_clean, weather_clean)
        
        results['collisions_clean'] = collisions_clean
        results['weather_clean'] = weather_clean
        results['merged'] = merged
        
        # Save silver data
        if not collisions_clean.empty:
            self.silver.save(collisions_clean, 'collisions')
        if not weather_clean.empty:
            self.silver.save(weather_clean, 'weather')
        if not merged.empty:
            self.silver.save(merged, 'merged')
        
        # Phase 4: Gold Layer
        print("\n" + "=" * 70)
        print("PHASE 4: GOLD LAYER (Analytics-Ready)")
        print("=" * 70)
        
        gold_microzone = self.gold.create_microzone_dataset(merged)
        results['gold_microzone'] = gold_microzone
        
        if not gold_microzone.empty:
            self.gold.save(gold_microzone, 'microzone')
            
            # Print summary
            summary = self.gold.get_zone_summary(gold_microzone)
            print("\n" + "-" * 50)
            print("GOLD LAYER SUMMARY")
            print("-" * 50)
            for key, value in summary.items():
                print(f"  {key}: {value}")
        
        # Pipeline complete
        elapsed = datetime.now() - start_time
        print("\n" + "=" * 70)
        print(f"PIPELINE COMPLETE - Elapsed time: {elapsed}")
        print("=" * 70)
        
        return results
    
    def run_incremental_update(self):
        """
        Run incremental update with latest data
        
        Returns:
            Updated gold microzone DataFrame
        """
        print("Running incremental update...")
        
        # Fetch latest data
        collisions_new, weather_dict = self.fetcher.fetch_realtime_update()
        
        if collisions_new.empty:
            print("No new data available")
            return None
        
        # Process through silver
        collisions_clean = self.silver.process_collisions(collisions_new)
        
        # Load existing gold and merge
        existing_gold = self.gold.load_latest()
        
        # Create new gold from incremental data
        # In production, this would merge with existing
        
        return collisions_clean
    
    def get_pipeline_status(self):
        """
        Get status of all pipeline layers
        
        Returns:
            Dictionary with layer statuses
        """
        return {
            'bronze': self.bronze.get_data_summary(),
            'silver': {
                'collisions': not self.silver.load_latest('collisions').empty,
                'weather': not self.silver.load_latest('weather').empty,
                'merged': not self.silver.load_latest('merged').empty
            },
            'gold': {
                'microzone': not self.gold.load_latest('microzone').empty
            }
        }


def main():
    """Main function to run the data pipeline"""
    pipeline = DataPipeline()
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(days_back=30, collision_limit=5000)
    
    # Display results
    print("\n" + "=" * 70)
    print("PIPELINE RESULTS")
    print("=" * 70)
    
    for name, df in results.items():
        if isinstance(df, pd.DataFrame):
            print(f"\n{name}:")
            print(f"  Records: {len(df)}")
            print(f"  Columns: {len(df.columns)}")
    
    return results


if __name__ == "__main__":
    main()
