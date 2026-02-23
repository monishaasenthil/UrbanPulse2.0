"""
Feature Engineer - Unified Feature Engineering Pipeline
Combines temporal, spatial, and environmental features
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.temporal_features import TemporalFeatureExtractor
from features.spatial_features import SpatialFeatureExtractor
from features.environmental_features import EnvironmentalFeatureExtractor


class FeatureEngineer:
    """
    Unified feature engineering pipeline for Urban Pulse
    
    Combines:
    - Temporal features (time-based patterns)
    - Spatial features (location-based)
    - Environmental features (weather-based)
    """
    
    def __init__(self):
        self.temporal = TemporalFeatureExtractor()
        self.spatial = SpatialFeatureExtractor()
        self.environmental = EnvironmentalFeatureExtractor()
        
        # Feature columns by category
        self.feature_groups = {
            'temporal': [],
            'spatial': [],
            'environmental': [],
            'target': ['severity', 'base_risk_score']
        }
        
    def engineer_features(self, df, gold_df=None, include_all=True):
        """
        Apply full feature engineering pipeline
        
        Args:
            df: Input DataFrame (merged collision + weather data)
            gold_df: Gold layer DataFrame for historical features
            include_all: Whether to include all feature types
            
        Returns:
            DataFrame with all engineered features
        """
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        
        result_df = df.copy()
        
        # 1. Temporal Features
        print("\n[1/4] Extracting temporal features...")
        result_df = self.temporal.extract_basic_temporal(result_df)
        result_df = self.temporal.extract_cyclical_features(result_df)
        result_df = self.temporal.add_context_labels(result_df)
        
        temporal_cols = ['hour', 'day_of_week', 'is_weekend', 'is_peak_hour', 
                        'is_night', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                        'context_label', 'time_period']
        self.feature_groups['temporal'] = [c for c in temporal_cols if c in result_df.columns]
        print(f"    Added {len(self.feature_groups['temporal'])} temporal features")
        
        # 2. Spatial Features
        print("\n[2/4] Extracting spatial features...")
        result_df = self.spatial.extract_hospital_proximity(result_df)
        result_df = self.spatial.extract_zone_density(result_df)
        result_df = self.spatial.add_major_road_indicator(result_df)
        
        if 'borough' in result_df.columns:
            result_df = self.spatial.extract_borough_features(result_df)
        
        if gold_df is not None and not gold_df.empty:
            result_df = self.spatial.extract_historical_risk(result_df, gold_df)
        
        spatial_cols = ['nearest_hospital_km', 'near_hospital', 'zone_density_score',
                       'is_major_road', 'historical_risk', 'borough_risk_factor']
        self.feature_groups['spatial'] = [c for c in spatial_cols if c in result_df.columns]
        print(f"    Added {len(self.feature_groups['spatial'])} spatial features")
        
        # 3. Environmental Features
        print("\n[3/4] Extracting environmental features...")
        result_df = self.environmental.extract_all_features(result_df)
        
        env_cols = ['is_raining', 'is_heavy_rain', 'rain_intensity_score',
                   'is_windy', 'wind_intensity_score', 'visibility_risk_score',
                   'temp_risk_score', 'weather_risk_score']
        self.feature_groups['environmental'] = [c for c in env_cols if c in result_df.columns]
        print(f"    Added {len(self.feature_groups['environmental'])} environmental features")
        
        # 4. Interaction Features
        print("\n[4/4] Creating interaction features...")
        result_df = self._create_interaction_features(result_df)
        
        print("\n" + "-" * 60)
        print("FEATURE ENGINEERING COMPLETE")
        print(f"Total features: {len(result_df.columns)}")
        print("-" * 60)
        
        return result_df
    
    def engineer_gold_features(self, gold_df):
        """
        Engineer features for gold layer (zone-level) data
        
        Args:
            gold_df: Gold microzone DataFrame
            
        Returns:
            DataFrame with zone-level features
        """
        if gold_df.empty:
            return gold_df
            
        result_df = gold_df.copy()
        
        # Add neighbor features
        result_df = self.spatial.extract_neighbor_features(result_df)
        
        # Add centrality
        result_df = self.spatial.calculate_zone_centrality(result_df)
        
        # Normalize numeric features
        numeric_cols = ['incident_count', 'total_severity', 'total_injured', 
                       'neighbor_avg_incidents', 'neighbor_avg_risk']
        
        for col in numeric_cols:
            if col in result_df.columns:
                result_df[f'{col}_normalized'] = self._normalize(result_df[col])
        
        return result_df
    
    def _create_interaction_features(self, df):
        """
        Create interaction features between different feature groups
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Peak hour + Rain interaction
        if 'is_peak_hour' in df.columns and 'is_raining' in df.columns:
            df['peak_rain_interaction'] = df['is_peak_hour'] * df['is_raining']
        
        # Night + Low visibility interaction
        if 'is_night' in df.columns and 'low_visibility' in df.columns:
            df['night_low_vis_interaction'] = df['is_night'] * df['low_visibility']
        
        # Weekend + Weather risk interaction
        if 'is_weekend' in df.columns and 'weather_risk_score' in df.columns:
            df['weekend_weather_interaction'] = df['is_weekend'] * df['weather_risk_score']
        
        # Major road + Peak hour interaction
        if 'is_major_road' in df.columns and 'is_peak_hour' in df.columns:
            df['major_road_peak_interaction'] = df['is_major_road'] * df['is_peak_hour']
        
        # Hospital proximity + Severity (for response time estimation)
        if 'nearest_hospital_km' in df.columns and 'severity' in df.columns:
            df['hospital_severity_interaction'] = df['nearest_hospital_km'] * df['severity']
        
        return df
    
    def get_feature_matrix(self, df, include_target=False):
        """
        Get feature matrix for ML models
        
        Args:
            df: DataFrame with all features
            include_target: Whether to include target columns
            
        Returns:
            Feature matrix (X) and optionally target (y)
        """
        # Collect all feature columns
        feature_cols = []
        for group, cols in self.feature_groups.items():
            if group != 'target':
                feature_cols.extend(cols)
        
        # Add interaction features
        interaction_cols = [c for c in df.columns if 'interaction' in c]
        feature_cols.extend(interaction_cols)
        
        # Filter to available columns
        available_cols = [c for c in feature_cols if c in df.columns]
        
        # Select numeric columns only
        X = df[available_cols].select_dtypes(include=[np.number])
        
        if include_target:
            target_cols = [c for c in self.feature_groups['target'] if c in df.columns]
            y = df[target_cols[0]] if target_cols else None
            return X, y
        
        return X
    
    def get_feature_names(self):
        """Get list of all feature names by group"""
        return self.feature_groups
    
    def _normalize(self, series):
        """Min-max normalize a series"""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)
    
    def prepare_ml_dataset(self, df, target_col='severity'):
        """
        Prepare dataset for ML training
        
        Args:
            df: DataFrame with all features
            target_col: Name of target column
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Get numeric features only
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Exclude target and metadata columns
        exclude_cols = [target_col, 'latitude', 'longitude', 'center_lat', 'center_lon',
                       'persons_injured', 'persons_killed', 'zone_incident_count']
        
        feature_cols = [c for c in numeric_df.columns if c not in exclude_cols]
        
        X = numeric_df[feature_cols].fillna(0)
        y = df[target_col] if target_col in df.columns else None
        
        return X, y, feature_cols


def main():
    """Test feature engineering pipeline"""
    # Create sample data
    sample_data = pd.DataFrame({
        'crash_datetime': pd.date_range('2024-01-01', periods=100, freq='H'),
        'latitude': np.random.uniform(40.6, 40.9, 100),
        'longitude': np.random.uniform(-74.1, -73.8, 100),
        'severity': np.random.randint(1, 10, 100),
        'persons_injured': np.random.randint(0, 5, 100),
        'persons_killed': np.random.randint(0, 2, 100),
        'borough': np.random.choice(['MANHATTAN', 'BROOKLYN', 'QUEENS'], 100),
        'temperature': np.random.uniform(0, 35, 100),
        'precipitation': np.random.exponential(1, 100),
        'wind_speed': np.random.uniform(0, 50, 100),
        'visibility': np.random.uniform(100, 10000, 100),
        'humidity': np.random.uniform(20, 100, 100)
    })
    
    # Add H3 index
    from data_engineering.h3_processor import H3Processor
    h3_proc = H3Processor()
    sample_data = h3_proc.add_h3_column(sample_data)
    
    # Run feature engineering
    engineer = FeatureEngineer()
    result = engineer.engineer_features(sample_data)
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING RESULTS")
    print("=" * 60)
    print(f"\nInput columns: {len(sample_data.columns)}")
    print(f"Output columns: {len(result.columns)}")
    print(f"\nFeature groups:")
    for group, cols in engineer.get_feature_names().items():
        print(f"  {group}: {len(cols)} features")
    
    # Prepare ML dataset
    X, y, feature_names = engineer.prepare_ml_dataset(result)
    print(f"\nML Dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape if y is not None else 'N/A'}")
    
    return result


if __name__ == "__main__":
    main()
