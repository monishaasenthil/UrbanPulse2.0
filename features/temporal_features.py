"""
Temporal Feature Extractor
Extracts time-based features for risk prediction
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CONTEXT_WINDOWS


class TemporalFeatureExtractor:
    """
    Extracts temporal features from collision data
    
    Features:
    - Hour of day
    - Day of week
    - Is weekend
    - Is peak hour (morning/evening)
    - Rolling incident counts (1hr, 6hr, 24hr)
    - Historical patterns
    """
    
    def __init__(self):
        self.context_windows = CONTEXT_WINDOWS
        
    def extract_basic_temporal(self, df, datetime_col='crash_datetime'):
        """
        Extract basic temporal features
        
        Args:
            df: DataFrame with datetime column
            datetime_col: Name of datetime column
            
        Returns:
            DataFrame with temporal features added
        """
        df = df.copy()
        
        if datetime_col not in df.columns:
            return df
            
        dt = pd.to_datetime(df[datetime_col])
        
        # Basic time features
        df['hour'] = dt.dt.hour
        df['minute'] = dt.dt.minute
        df['day_of_week'] = dt.dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = dt.dt.day
        df['month'] = dt.dt.month
        df['year'] = dt.dt.year
        df['week_of_year'] = dt.dt.isocalendar().week
        
        # Derived features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_weekday'] = (~df['day_of_week'].isin([5, 6])).astype(int)
        
        # Peak hour indicators
        df['is_morning_peak'] = df['hour'].isin([7, 8, 9]).astype(int)
        df['is_evening_peak'] = df['hour'].isin([16, 17, 18, 19]).astype(int)
        df['is_peak_hour'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)
        
        # Night indicator
        df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        
        # Time period categories
        df['time_period'] = df['hour'].apply(self._get_time_period)
        
        return df
    
    def extract_cyclical_features(self, df):
        """
        Extract cyclical temporal features using sin/cos encoding
        
        Args:
            df: DataFrame with hour and day_of_week columns
            
        Returns:
            DataFrame with cyclical features added
        """
        df = df.copy()
        
        # Hour cyclical encoding (24-hour cycle)
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week cyclical encoding (7-day cycle)
        if 'day_of_week' in df.columns:
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month cyclical encoding (12-month cycle)
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def extract_rolling_features(self, df, h3_col='h3_index', datetime_col='crash_datetime',
                                  windows=['1H', '6H', '24H', '7D']):
        """
        Extract rolling window features per H3 zone
        
        Args:
            df: DataFrame with H3 and datetime columns
            h3_col: Name of H3 index column
            datetime_col: Name of datetime column
            windows: List of time windows for rolling calculations
            
        Returns:
            DataFrame with rolling features
        """
        if df.empty or h3_col not in df.columns:
            return df
            
        df = df.copy()
        df = df.sort_values(datetime_col)
        
        # Group by H3 zone and calculate rolling features
        rolling_features = []
        
        for h3_idx in df[h3_col].unique():
            zone_df = df[df[h3_col] == h3_idx].copy()
            zone_df = zone_df.set_index(datetime_col)
            
            for window in windows:
                # Rolling incident count
                col_name = f'incidents_last_{window}'
                zone_df[col_name] = zone_df['severity'].rolling(window, min_periods=1).count()
                
                # Rolling severity sum
                col_name = f'severity_last_{window}'
                zone_df[col_name] = zone_df['severity'].rolling(window, min_periods=1).sum()
            
            zone_df = zone_df.reset_index()
            rolling_features.append(zone_df)
        
        if rolling_features:
            return pd.concat(rolling_features, ignore_index=True)
        return df
    
    def extract_lag_features(self, df, h3_col='h3_index', target_col='severity',
                             lags=[1, 2, 3, 6, 12, 24]):
        """
        Extract lag features for time series modeling
        
        Args:
            df: DataFrame sorted by time
            h3_col: Name of H3 index column
            target_col: Column to create lags for
            lags: List of lag periods (in hours)
            
        Returns:
            DataFrame with lag features
        """
        if df.empty or h3_col not in df.columns:
            return df
            
        df = df.copy()
        
        for lag in lags:
            col_name = f'{target_col}_lag_{lag}h'
            df[col_name] = df.groupby(h3_col)[target_col].shift(lag)
        
        return df
    
    def get_context_label(self, hour, is_raining=False):
        """
        Get context label for a given hour and weather
        
        Args:
            hour: Hour of day (0-23)
            is_raining: Whether it's raining
            
        Returns:
            Context label string
        """
        if is_raining:
            return 'rainy'
        
        morning_start, morning_end = self.context_windows['morning_peak']
        evening_start, evening_end = self.context_windows['evening_peak']
        night_start, night_end = self.context_windows['night']
        
        if morning_start <= hour < morning_end:
            return 'morning_peak'
        elif evening_start <= hour < evening_end:
            return 'evening_peak'
        elif hour >= night_start or hour < night_end:
            return 'night'
        else:
            return 'normal'
    
    def add_context_labels(self, df, hour_col='hour', rain_col='is_raining'):
        """
        Add context labels to DataFrame
        
        Args:
            df: DataFrame with hour and rain columns
            hour_col: Name of hour column
            rain_col: Name of rain indicator column
            
        Returns:
            DataFrame with context_label column
        """
        df = df.copy()
        
        is_raining = df[rain_col] if rain_col in df.columns else False
        
        df['context_label'] = df.apply(
            lambda row: self.get_context_label(
                row[hour_col] if hour_col in df.columns else 12,
                row[rain_col] if rain_col in df.columns else False
            ),
            axis=1
        )
        
        return df
    
    def _get_time_period(self, hour):
        """Categorize hour into time period"""
        if 6 <= hour < 10:
            return 'morning'
        elif 10 <= hour < 14:
            return 'midday'
        elif 14 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    def create_hourly_profile(self, df, h3_col='h3_index'):
        """
        Create hourly incident profile for each zone
        
        Args:
            df: DataFrame with H3 and hour columns
            h3_col: Name of H3 index column
            
        Returns:
            DataFrame with hourly profile features
        """
        if df.empty or 'hour' not in df.columns:
            return pd.DataFrame()
        
        # Pivot to get incidents by hour
        hourly = df.groupby([h3_col, 'hour']).size().unstack(fill_value=0)
        
        # Normalize to get proportions
        hourly_norm = hourly.div(hourly.sum(axis=1), axis=0).fillna(0)
        
        # Rename columns
        hourly_norm.columns = [f'hour_{h}_ratio' for h in hourly_norm.columns]
        
        return hourly_norm.reset_index()
