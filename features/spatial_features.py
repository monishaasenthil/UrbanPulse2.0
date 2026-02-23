"""
Spatial Feature Extractor
Extracts location-based features for risk prediction
"""
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_engineering.h3_processor import H3Processor


class SpatialFeatureExtractor:
    """
    Extracts spatial features from location data
    
    Features:
    - Proximity to hospitals
    - Proximity to major roads
    - Historical risk by zone
    - Neighbor zone characteristics
    - Zone density metrics
    """
    
    def __init__(self):
        self.h3_processor = H3Processor()
        
        # NYC Hospital locations (sample - major hospitals)
        self.hospitals = [
            {"name": "NYU Langone", "lat": 40.7421, "lon": -73.9739},
            {"name": "Mount Sinai", "lat": 40.7900, "lon": -73.9526},
            {"name": "NY Presbyterian", "lat": 40.8404, "lon": -73.9419},
            {"name": "Bellevue Hospital", "lat": 40.7392, "lon": -73.9750},
            {"name": "Brooklyn Hospital", "lat": 40.6892, "lon": -73.9784},
            {"name": "Jamaica Hospital", "lat": 40.7033, "lon": -73.8167},
            {"name": "Staten Island Univ Hospital", "lat": 40.5834, "lon": -74.0932},
            {"name": "Lincoln Medical Center", "lat": 40.8168, "lon": -73.9249},
        ]
        
        # Major road indicators (simplified - based on known busy areas)
        self.major_road_zones = set()  # Will be populated from data
        
    def extract_hospital_proximity(self, df, lat_col='latitude', lon_col='longitude'):
        """
        Calculate proximity to nearest hospital
        
        Args:
            df: DataFrame with lat/lon columns
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            
        Returns:
            DataFrame with hospital proximity features
        """
        df = df.copy()
        
        def get_nearest_hospital_distance(lat, lon):
            if pd.isna(lat) or pd.isna(lon):
                return np.nan, None
            
            min_dist = float('inf')
            nearest = None
            
            for hospital in self.hospitals:
                dist = self._haversine_distance(lat, lon, hospital['lat'], hospital['lon'])
                if dist < min_dist:
                    min_dist = dist
                    nearest = hospital['name']
            
            return min_dist, nearest
        
        # Calculate for each row
        results = df.apply(
            lambda row: get_nearest_hospital_distance(row[lat_col], row[lon_col]),
            axis=1
        )
        
        df['nearest_hospital_km'] = results.apply(lambda x: x[0])
        df['nearest_hospital_name'] = results.apply(lambda x: x[1])
        
        # Create proximity categories
        df['hospital_proximity'] = pd.cut(
            df['nearest_hospital_km'],
            bins=[0, 1, 3, 5, 10, float('inf')],
            labels=['very_close', 'close', 'moderate', 'far', 'very_far']
        )
        
        # Binary indicator for close to hospital
        df['near_hospital'] = (df['nearest_hospital_km'] < 2).astype(int)
        
        return df
    
    def extract_zone_density(self, df, h3_col='h3_index'):
        """
        Calculate incident density per zone
        
        Args:
            df: DataFrame with H3 index
            h3_col: Name of H3 column
            
        Returns:
            DataFrame with zone density features
        """
        if df.empty or h3_col not in df.columns:
            return df
            
        df = df.copy()
        
        # Count incidents per zone
        zone_counts = df[h3_col].value_counts()
        
        # Map back to dataframe
        df['zone_incident_count'] = df[h3_col].map(zone_counts)
        
        # Normalize to density score
        max_count = zone_counts.max()
        df['zone_density_score'] = df['zone_incident_count'] / max_count if max_count > 0 else 0
        
        # Categorize density
        df['zone_density_category'] = pd.cut(
            df['zone_density_score'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        return df
    
    def extract_neighbor_features(self, gold_df, h3_col='h3_index'):
        """
        Extract features from neighboring zones
        
        Args:
            gold_df: Gold layer DataFrame with zone aggregates
            h3_col: Name of H3 column
            
        Returns:
            DataFrame with neighbor features
        """
        if gold_df.empty or h3_col not in gold_df.columns:
            return gold_df
            
        df = gold_df.copy()
        
        # Create lookup for zone metrics
        zone_metrics = df.set_index(h3_col)[['incident_count', 'base_risk_score']].to_dict('index')
        
        def get_neighbor_stats(h3_idx):
            neighbors = self.h3_processor.get_neighbors(h3_idx, ring_size=1)
            neighbors = neighbors - {h3_idx}  # Exclude self
            
            neighbor_incidents = []
            neighbor_risks = []
            
            for n in neighbors:
                if n in zone_metrics:
                    neighbor_incidents.append(zone_metrics[n].get('incident_count', 0))
                    neighbor_risks.append(zone_metrics[n].get('base_risk_score', 0))
            
            return {
                'neighbor_count': len(neighbors),
                'neighbor_avg_incidents': np.mean(neighbor_incidents) if neighbor_incidents else 0,
                'neighbor_max_incidents': max(neighbor_incidents) if neighbor_incidents else 0,
                'neighbor_avg_risk': np.mean(neighbor_risks) if neighbor_risks else 0,
                'neighbor_max_risk': max(neighbor_risks) if neighbor_risks else 0,
                'high_risk_neighbors': sum(1 for r in neighbor_risks if r > 0.6)
            }
        
        # Apply to each zone
        neighbor_stats = df[h3_col].apply(get_neighbor_stats)
        neighbor_df = pd.DataFrame(neighbor_stats.tolist())
        
        # Combine with original
        for col in neighbor_df.columns:
            df[col] = neighbor_df[col].values
        
        return df
    
    def extract_historical_risk(self, df, gold_df, h3_col='h3_index'):
        """
        Add historical risk score from gold layer
        
        Args:
            df: DataFrame to add features to
            gold_df: Gold layer DataFrame with historical risk
            h3_col: Name of H3 column
            
        Returns:
            DataFrame with historical risk features
        """
        if df.empty or gold_df.empty:
            return df
            
        df = df.copy()
        
        # Create risk lookup
        risk_lookup = gold_df.set_index(h3_col)['base_risk_score'].to_dict()
        
        df['historical_risk'] = df[h3_col].map(risk_lookup).fillna(0)
        
        # Risk category
        df['historical_risk_category'] = pd.cut(
            df['historical_risk'],
            bins=[-np.inf, 0.3, 0.6, 0.8, np.inf],
            labels=['low', 'medium', 'high', 'critical']
        )
        
        return df
    
    def identify_major_roads(self, df, threshold_percentile=80):
        """
        Identify major roads based on incident density
        
        Args:
            df: DataFrame with H3 indices
            threshold_percentile: Percentile threshold for major roads
        """
        if 'h3_index' not in df.columns:
            return
        
        zone_counts = df['h3_index'].value_counts()
        if len(zone_counts) == 0:
            self.major_road_zones = set()
            return
        threshold = np.percentile(zone_counts.values, threshold_percentile)
        
        major_road_zones = set(zone_counts[zone_counts >= threshold].index)
        self.major_road_zones = major_road_zones
        
        return major_road_zones
    
    def add_major_road_indicator(self, df, h3_col='h3_index'):
        """
        Add major road indicator to DataFrame
        
        Args:
            df: DataFrame with H3 column
            h3_col: Name of H3 column
            
        Returns:
            DataFrame with major_road indicator
        """
        df = df.copy()
        
        if not self.major_road_zones:
            self.identify_major_roads(df)
        
        df['is_major_road'] = df[h3_col].isin(self.major_road_zones).astype(int)
        
        return df
    
    def extract_borough_features(self, df, borough_col='borough'):
        """
        Extract borough-based features
        
        Args:
            df: DataFrame with borough column
            borough_col: Name of borough column
            
        Returns:
            DataFrame with borough features
        """
        if df.empty or borough_col not in df.columns:
            return df
            
        df = df.copy()
        
        # Borough risk profiles (based on typical patterns)
        borough_risk = {
            'MANHATTAN': 0.8,
            'BROOKLYN': 0.7,
            'QUEENS': 0.6,
            'BRONX': 0.65,
            'STATEN ISLAND': 0.4,
            'UNKNOWN': 0.5
        }
        
        df['borough_risk_factor'] = df[borough_col].map(borough_risk).fillna(0.5)
        
        # One-hot encode boroughs
        borough_dummies = pd.get_dummies(df[borough_col], prefix='borough')
        df = pd.concat([df, borough_dummies], axis=1)
        
        return df
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate haversine distance between two points in km
        """
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def calculate_zone_centrality(self, gold_df, h3_col='h3_index'):
        """
        Calculate centrality metrics for each zone
        
        Args:
            gold_df: Gold layer DataFrame
            h3_col: Name of H3 column
            
        Returns:
            DataFrame with centrality features
        """
        if gold_df.empty:
            return gold_df
            
        df = gold_df.copy()
        
        # Calculate center of all zones
        center_lat = df['center_lat'].mean()
        center_lon = df['center_lon'].mean()
        
        # Distance from center
        df['distance_from_center'] = df.apply(
            lambda row: self._haversine_distance(
                row['center_lat'], row['center_lon'],
                center_lat, center_lon
            ),
            axis=1
        )
        
        # Normalize
        max_dist = df['distance_from_center'].max()
        df['centrality_score'] = 1 - (df['distance_from_center'] / max_dist) if max_dist > 0 else 0.5
        
        return df
