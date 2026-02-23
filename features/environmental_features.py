"""
Environmental Feature Extractor
Extracts weather and environmental features for risk prediction
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import WEATHER_THRESHOLDS


class EnvironmentalFeatureExtractor:
    """
    Extracts environmental features from weather data
    
    Features:
    - Rain intensity
    - Wind conditions
    - Temperature extremes
    - Visibility conditions
    - Combined weather risk score
    """
    
    def __init__(self):
        self.thresholds = WEATHER_THRESHOLDS
        
    def extract_precipitation_features(self, df, precip_col='precipitation'):
        """
        Extract precipitation-related features
        
        Args:
            df: DataFrame with precipitation column
            precip_col: Name of precipitation column
            
        Returns:
            DataFrame with precipitation features
        """
        df = df.copy()
        
        if precip_col not in df.columns:
            return df
        
        precip = df[precip_col].fillna(0)
        
        # Binary indicators
        df['is_raining'] = (precip > 0).astype(int)
        df['is_light_rain'] = ((precip > 0) & (precip <= self.thresholds['light_rain'])).astype(int)
        df['is_moderate_rain'] = ((precip > self.thresholds['light_rain']) & 
                                   (precip <= self.thresholds['heavy_rain'])).astype(int)
        df['is_heavy_rain'] = (precip > self.thresholds['heavy_rain']).astype(int)
        
        # Intensity category
        df['rain_intensity'] = pd.cut(
            precip,
            bins=[-np.inf, 0, 0.5, 2.5, 7.5, np.inf],
            labels=['none', 'light', 'moderate', 'heavy', 'extreme']
        )
        
        # Numeric intensity score (0-1)
        max_precip = max(precip.max(), 10)  # Cap at 10mm for normalization
        df['rain_intensity_score'] = (precip / max_precip).clip(0, 1)
        
        return df
    
    def extract_wind_features(self, df, wind_col='wind_speed', direction_col='wind_direction'):
        """
        Extract wind-related features
        
        Args:
            df: DataFrame with wind columns
            wind_col: Name of wind speed column
            direction_col: Name of wind direction column
            
        Returns:
            DataFrame with wind features
        """
        df = df.copy()
        
        if wind_col not in df.columns:
            return df
        
        wind = df[wind_col].fillna(10)  # Default moderate wind
        
        # Binary indicators
        df['is_calm'] = (wind < 10).astype(int)
        df['is_windy'] = (wind > self.thresholds['high_wind']).astype(int)
        df['is_very_windy'] = (wind > 50).astype(int)
        
        # Wind category
        df['wind_category'] = pd.cut(
            wind,
            bins=[-np.inf, 10, 20, 40, 60, np.inf],
            labels=['calm', 'light', 'moderate', 'strong', 'severe']
        )
        
        # Numeric intensity score
        max_wind = max(wind.max(), 60)
        df['wind_intensity_score'] = (wind / max_wind).clip(0, 1)
        
        # Wind direction features (if available)
        if direction_col in df.columns:
            direction = df[direction_col].fillna(0)
            
            # Convert to cardinal direction
            df['wind_cardinal'] = direction.apply(self._get_cardinal_direction)
            
            # Cyclical encoding
            df['wind_dir_sin'] = np.sin(np.radians(direction))
            df['wind_dir_cos'] = np.cos(np.radians(direction))
        
        return df
    
    def extract_visibility_features(self, df, vis_col='visibility'):
        """
        Extract visibility-related features
        
        Args:
            df: DataFrame with visibility column
            vis_col: Name of visibility column (in meters)
            
        Returns:
            DataFrame with visibility features
        """
        df = df.copy()
        
        if vis_col not in df.columns:
            return df
        
        vis = df[vis_col].fillna(10000)  # Default good visibility
        
        # Binary indicators
        df['low_visibility'] = (vis < self.thresholds['low_visibility']).astype(int)
        df['very_low_visibility'] = (vis < 500).astype(int)
        df['good_visibility'] = (vis > 5000).astype(int)
        
        # Visibility category
        df['visibility_category'] = pd.cut(
            vis,
            bins=[-np.inf, 200, 500, 1000, 5000, np.inf],
            labels=['very_poor', 'poor', 'moderate', 'good', 'excellent']
        )
        
        # Inverse visibility score (higher = worse visibility = higher risk)
        df['visibility_risk_score'] = (1 - vis / 10000).clip(0, 1)
        
        return df
    
    def extract_temperature_features(self, df, temp_col='temperature'):
        """
        Extract temperature-related features
        
        Args:
            df: DataFrame with temperature column
            temp_col: Name of temperature column (in Celsius)
            
        Returns:
            DataFrame with temperature features
        """
        df = df.copy()
        
        if temp_col not in df.columns:
            return df
        
        temp = df[temp_col].fillna(20)  # Default moderate temp
        
        # Extreme temperature indicators
        df['is_freezing'] = (temp <= 0).astype(int)
        df['is_cold'] = ((temp > 0) & (temp < 10)).astype(int)
        df['is_hot'] = (temp > 30).astype(int)
        df['is_extreme_temp'] = ((temp <= 0) | (temp > 35)).astype(int)
        
        # Temperature category
        df['temp_category'] = pd.cut(
            temp,
            bins=[-np.inf, 0, 10, 20, 30, np.inf],
            labels=['freezing', 'cold', 'mild', 'warm', 'hot']
        )
        
        # Temperature risk score (extreme temps = higher risk)
        # Optimal around 15-25°C
        df['temp_risk_score'] = df[temp_col].apply(self._temp_risk_score)
        
        return df
    
    def extract_humidity_features(self, df, humidity_col='humidity'):
        """
        Extract humidity-related features
        
        Args:
            df: DataFrame with humidity column
            humidity_col: Name of humidity column (percentage)
            
        Returns:
            DataFrame with humidity features
        """
        df = df.copy()
        
        if humidity_col not in df.columns:
            return df
        
        humidity = df[humidity_col].fillna(50)
        
        # Humidity indicators
        df['is_humid'] = (humidity > 80).astype(int)
        df['is_dry'] = (humidity < 30).astype(int)
        
        # Humidity category
        df['humidity_category'] = pd.cut(
            humidity,
            bins=[-np.inf, 30, 50, 70, 85, np.inf],
            labels=['very_dry', 'dry', 'comfortable', 'humid', 'very_humid']
        )
        
        return df
    
    def calculate_weather_risk_score(self, df):
        """
        Calculate combined weather risk score
        
        Args:
            df: DataFrame with individual weather features
            
        Returns:
            DataFrame with combined weather_risk_score
        """
        df = df.copy()
        
        # Component scores (default to 0 if not available)
        rain_score = df.get('rain_intensity_score', pd.Series([0] * len(df)))
        wind_score = df.get('wind_intensity_score', pd.Series([0] * len(df)))
        vis_score = df.get('visibility_risk_score', pd.Series([0] * len(df)))
        temp_score = df.get('temp_risk_score', pd.Series([0] * len(df)))
        
        # Weighted combination
        weights = {
            'rain': 0.35,
            'wind': 0.25,
            'visibility': 0.25,
            'temperature': 0.15
        }
        
        df['weather_risk_score'] = (
            weights['rain'] * rain_score +
            weights['wind'] * wind_score +
            weights['visibility'] * vis_score +
            weights['temperature'] * temp_score
        ).clip(0, 1)
        
        # Weather risk category
        df['weather_risk_category'] = pd.cut(
            df['weather_risk_score'],
            bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
            labels=['minimal', 'low', 'moderate', 'high', 'severe']
        )
        
        return df
    
    def extract_all_features(self, df):
        """
        Extract all environmental features
        
        Args:
            df: DataFrame with weather columns
            
        Returns:
            DataFrame with all environmental features
        """
        df = self.extract_precipitation_features(df)
        df = self.extract_wind_features(df)
        df = self.extract_visibility_features(df)
        df = self.extract_temperature_features(df)
        df = self.extract_humidity_features(df)
        df = self.calculate_weather_risk_score(df)
        
        return df
    
    def _get_cardinal_direction(self, degrees):
        """Convert degrees to cardinal direction"""
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        idx = round(degrees / 45) % 8
        return directions[idx]
    
    def _temp_risk_score(self, temp):
        """Calculate temperature risk score (0-1)"""
        if pd.isna(temp):
            return 0.2
        
        # Optimal range: 15-25°C
        if 15 <= temp <= 25:
            return 0.1
        elif 10 <= temp < 15 or 25 < temp <= 30:
            return 0.3
        elif 5 <= temp < 10 or 30 < temp <= 35:
            return 0.5
        elif 0 <= temp < 5 or 35 < temp <= 40:
            return 0.7
        else:  # Below 0 or above 40
            return 0.9
    
    def get_weather_summary(self, df):
        """
        Get summary statistics for weather features
        
        Args:
            df: DataFrame with weather features
            
        Returns:
            Dictionary with weather summary
        """
        summary = {}
        
        if 'is_raining' in df.columns:
            summary['rain_percentage'] = df['is_raining'].mean() * 100
        
        if 'is_windy' in df.columns:
            summary['windy_percentage'] = df['is_windy'].mean() * 100
        
        if 'low_visibility' in df.columns:
            summary['low_vis_percentage'] = df['low_visibility'].mean() * 100
        
        if 'weather_risk_score' in df.columns:
            summary['avg_weather_risk'] = df['weather_risk_score'].mean()
            summary['max_weather_risk'] = df['weather_risk_score'].max()
        
        return summary
