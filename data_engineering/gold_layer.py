"""
Gold Layer - Analytics-Ready Aggregated Data
Creates the final gold_microzone_dataset for ML and decision making
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import GOLD_DIR
from data_engineering.h3_processor import H3Processor


class GoldLayer:
    """
    Gold Layer for analytics-ready aggregated data
    - Aggregates by H3 micro-zone
    - Creates temporal windows
    - Generates ML-ready features
    """
    
    def __init__(self):
        self.gold_dir = GOLD_DIR
        os.makedirs(self.gold_dir, exist_ok=True)
        self.h3_processor = H3Processor()
        
    def create_microzone_dataset(self, merged_df):
        """
        Create the gold microzone dataset aggregated by H3 hexagon
        
        Args:
            merged_df: Merged collision+weather data from silver layer
            
        Returns:
            Gold microzone DataFrame
        """
        if merged_df.empty:
            return pd.DataFrame()
        
        print("\nCreating Gold Microzone Dataset...")
        
        # Filter valid H3 records
        df = merged_df[merged_df['h3_index'].notna()].copy()
        print(f"  Valid H3 records: {len(df)}")
        
        # Aggregate by H3 hexagon
        agg_dict = {
            # Incident metrics
            'severity': ['sum', 'mean', 'max'],
            'persons_injured': 'sum',
            'persons_killed': 'sum',
            
            # Count
            'crash_datetime': 'count',
            
            # Weather averages (if available)
            'temperature': 'mean',
            'humidity': 'mean',
            'precipitation': 'mean',
            'wind_speed': 'mean',
            
            # Boolean aggregates
            'is_raining': 'mean',  # Proportion of rainy incidents
            'is_peak_hour': 'mean',  # Proportion during peak hours
            'is_weekend': 'mean',  # Proportion on weekends
        }
        
        # Filter to available columns
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        gold_df = df.groupby('h3_index').agg(agg_dict).reset_index()
        
        # Flatten column names
        gold_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                          for col in gold_df.columns]
        
        # Rename for clarity
        rename_map = {
            'crash_datetime_count': 'incident_count',
            'severity_sum': 'total_severity',
            'severity_mean': 'avg_severity',
            'severity_max': 'max_severity',
            'persons_injured_sum': 'total_injured',
            'persons_killed_sum': 'total_killed',
            'temperature_mean': 'avg_temperature',
            'humidity_mean': 'avg_humidity',
            'precipitation_mean': 'avg_precipitation',
            'wind_speed_mean': 'avg_wind_speed',
            'is_raining_mean': 'rain_incident_ratio',
            'is_peak_hour_mean': 'peak_hour_ratio',
            'is_weekend_mean': 'weekend_ratio'
        }
        gold_df = gold_df.rename(columns={k: v for k, v in rename_map.items() if k in gold_df.columns})
        
        # Add H3 center coordinates
        gold_df['center_lat'] = gold_df['h3_index'].apply(
            lambda x: self.h3_processor.h3_to_center(x)[0]
        )
        gold_df['center_lon'] = gold_df['h3_index'].apply(
            lambda x: self.h3_processor.h3_to_center(x)[1]
        )
        
        # Calculate base risk score
        gold_df = self._calculate_base_risk(gold_df)
        
        # Add neighbor count
        gold_df['neighbor_count'] = gold_df['h3_index'].apply(
            lambda x: len(self.h3_processor.get_neighbors(x, ring_size=1)) - 1
        )
        
        print(f"  Created {len(gold_df)} microzone records")
        
        return gold_df
    
    def create_temporal_aggregates(self, merged_df, time_windows=['1H', '6H', '24H']):
        """
        Create temporal aggregates for each H3 zone
        
        Args:
            merged_df: Merged data from silver layer
            time_windows: List of time window sizes
            
        Returns:
            DataFrame with temporal features
        """
        if merged_df.empty:
            return pd.DataFrame()
        
        df = merged_df[merged_df['h3_index'].notna()].copy()
        df = df.sort_values('crash_datetime')
        
        results = []
        
        for h3_idx in df['h3_index'].unique():
            zone_df = df[df['h3_index'] == h3_idx].copy()
            zone_df = zone_df.set_index('crash_datetime')
            
            record = {'h3_index': h3_idx}
            
            for window in time_windows:
                # Rolling incident count
                rolling = zone_df['severity'].rolling(window).sum()
                record[f'incidents_{window}'] = rolling.iloc[-1] if len(rolling) > 0 else 0
                
            results.append(record)
        
        return pd.DataFrame(results)
    
    def create_hourly_patterns(self, merged_df):
        """
        Create hourly incident patterns for each zone
        
        Args:
            merged_df: Merged data from silver layer
            
        Returns:
            DataFrame with hourly patterns
        """
        if merged_df.empty or 'hour' not in merged_df.columns:
            return pd.DataFrame()
        
        df = merged_df[merged_df['h3_index'].notna()].copy()
        
        # Pivot to get incidents by hour for each zone
        hourly = df.groupby(['h3_index', 'hour']).size().unstack(fill_value=0)
        hourly.columns = [f'hour_{h}_incidents' for h in hourly.columns]
        hourly = hourly.reset_index()
        
        # Add peak hour indicators
        morning_cols = [f'hour_{h}_incidents' for h in [7, 8, 9] if f'hour_{h}_incidents' in hourly.columns]
        evening_cols = [f'hour_{h}_incidents' for h in [16, 17, 18, 19] if f'hour_{h}_incidents' in hourly.columns]
        
        if morning_cols:
            hourly['morning_peak_incidents'] = hourly[morning_cols].sum(axis=1)
        if evening_cols:
            hourly['evening_peak_incidents'] = hourly[evening_cols].sum(axis=1)
        
        return hourly
    
    def _calculate_base_risk(self, gold_df):
        """
        Calculate base risk score for each microzone
        
        Args:
            gold_df: Gold layer DataFrame
            
        Returns:
            DataFrame with base_risk_score added
        """
        df = gold_df.copy()
        
        # Normalize components
        if 'incident_count' in df.columns:
            df['norm_incidents'] = self._normalize(df['incident_count'])
        else:
            df['norm_incidents'] = 0
            
        if 'total_severity' in df.columns:
            df['norm_severity'] = self._normalize(df['total_severity'])
        else:
            df['norm_severity'] = 0
            
        if 'total_killed' in df.columns:
            df['norm_fatalities'] = self._normalize(df['total_killed'])
        else:
            df['norm_fatalities'] = 0
        
        # Calculate base risk (weighted combination)
        df['base_risk_score'] = (
            0.3 * df['norm_incidents'] +
            0.4 * df['norm_severity'] +
            0.3 * df['norm_fatalities']
        )
        
        # Clip to [0, 1]
        df['base_risk_score'] = df['base_risk_score'].clip(0, 1)
        
        # Add risk category
        df['risk_category'] = pd.cut(
            df['base_risk_score'],
            bins=[-np.inf, 0.3, 0.6, 0.8, np.inf],
            labels=['low', 'medium', 'high', 'critical']
        )
        
        # Clean up temp columns
        df = df.drop(columns=['norm_incidents', 'norm_severity', 'norm_fatalities'], errors='ignore')
        
        return df
    
    def _normalize(self, series):
        """Min-max normalize a series to [0, 1]"""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)
    
    def save(self, df, name="microzone"):
        """
        Save gold layer data
        
        Args:
            df: Gold DataFrame
            name: Dataset name
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gold_{name}_{timestamp}.csv"
        filepath = os.path.join(self.gold_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"Gold Layer: Saved {len(df)} records to {filename}")
        
        return filepath
    
    def load_latest(self, name="microzone"):
        """Load most recent gold data"""
        prefix = f"gold_{name}"
        files = [f for f in os.listdir(self.gold_dir) 
                if f.startswith(prefix) and f.endswith('.csv')]
        
        if not files:
            return pd.DataFrame()
        
        latest = max(files)
        return pd.read_csv(os.path.join(self.gold_dir, latest))
    
    def get_zone_summary(self, gold_df):
        """
        Get summary statistics for gold dataset
        
        Args:
            gold_df: Gold microzone DataFrame
            
        Returns:
            Dictionary with summary stats
        """
        if gold_df.empty:
            return {}
        
        return {
            'total_zones': len(gold_df),
            'total_incidents': gold_df.get('incident_count', pd.Series([0])).sum(),
            'total_injured': gold_df.get('total_injured', pd.Series([0])).sum(),
            'total_killed': gold_df.get('total_killed', pd.Series([0])).sum(),
            'avg_risk_score': gold_df.get('base_risk_score', pd.Series([0])).mean(),
            'high_risk_zones': (gold_df.get('risk_category', pd.Series()) == 'high').sum(),
            'critical_zones': (gold_df.get('risk_category', pd.Series()) == 'critical').sum()
        }
