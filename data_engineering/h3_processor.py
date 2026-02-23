"""
H3 Micro-Zoning Processor
Converts geographic coordinates to H3 hexagonal indices
"""
import h3
import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import H3_RESOLUTION


class H3Processor:
    """
    Processor for H3 hexagonal spatial indexing
    Enables micro-zone level intelligence for Urban Pulse
    """
    
    def __init__(self, resolution=H3_RESOLUTION):
        """
        Initialize H3 processor
        
        Args:
            resolution: H3 resolution level (0-15, default 8)
                       Resolution 8 ≈ 0.74 km² per hexagon
        """
        self.resolution = resolution
        
    def lat_lon_to_h3(self, latitude, longitude):
        """
        Convert a single lat/lon pair to H3 index
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            H3 index string
        """
        try:
            if pd.isna(latitude) or pd.isna(longitude):
                return None
            # H3 v4.x uses latlng_to_cell instead of geo_to_h3
            return h3.latlng_to_cell(latitude, longitude, self.resolution)
        except Exception:
            return None
    
    def add_h3_column(self, df, lat_col='latitude', lon_col='longitude', h3_col='h3_index'):
        """
        Add H3 index column to DataFrame
        
        Args:
            df: Input DataFrame
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            h3_col: Name for new H3 column
            
        Returns:
            DataFrame with H3 column added
        """
        df = df.copy()
        df[h3_col] = df.apply(
            lambda row: self.lat_lon_to_h3(row[lat_col], row[lon_col]),
            axis=1
        )
        return df
    
    def h3_to_center(self, h3_index):
        """
        Get center coordinates of H3 hexagon
        
        Args:
            h3_index: H3 index string
            
        Returns:
            Tuple of (latitude, longitude)
        """
        try:
            # H3 v4.x uses cell_to_latlng instead of h3_to_geo
            return h3.cell_to_latlng(h3_index)
        except Exception:
            return (None, None)
    
    def get_neighbors(self, h3_index, ring_size=1):
        """
        Get neighboring hexagons
        
        Args:
            h3_index: Center H3 index
            ring_size: Number of rings of neighbors (1 = immediate neighbors)
            
        Returns:
            Set of neighboring H3 indices
        """
        try:
            # H3 v4.x uses grid_disk instead of k_ring
            return h3.grid_disk(h3_index, ring_size)
        except Exception:
            return set()
    
    def get_neighbor_ring(self, h3_index, ring_distance):
        """
        Get hexagons at exactly ring_distance from center
        
        Args:
            h3_index: Center H3 index
            ring_distance: Distance in rings
            
        Returns:
            Set of H3 indices at that distance
        """
        try:
            # H3 v4.x uses grid_ring instead of hex_ring
            return h3.grid_ring(h3_index, ring_distance)
        except Exception:
            return set()
    
    def aggregate_by_h3(self, df, h3_col='h3_index', agg_config=None):
        """
        Aggregate data by H3 hexagon
        
        Args:
            df: Input DataFrame with H3 column
            h3_col: Name of H3 column
            agg_config: Dictionary of {column: aggregation_function}
            
        Returns:
            Aggregated DataFrame by H3 index
        """
        if agg_config is None:
            # Default aggregation for collision data
            agg_config = {
                'severity': 'sum',
                'persons_injured': 'sum',
                'persons_killed': 'sum',
                'crash_datetime': 'count'  # Count of incidents
            }
        
        # Filter out rows without H3 index
        df_valid = df[df[h3_col].notna()].copy()
        
        # Perform aggregation
        agg_df = df_valid.groupby(h3_col).agg(agg_config).reset_index()
        
        # Rename count column
        if 'crash_datetime' in agg_config:
            agg_df = agg_df.rename(columns={'crash_datetime': 'incident_count'})
        
        # Add center coordinates
        agg_df['center_lat'] = agg_df[h3_col].apply(lambda x: self.h3_to_center(x)[0])
        agg_df['center_lon'] = agg_df[h3_col].apply(lambda x: self.h3_to_center(x)[1])
        
        return agg_df
    
    def create_h3_grid(self, bounds, resolution=None):
        """
        Create H3 grid covering a bounding box
        
        Args:
            bounds: Dictionary with 'min_lat', 'max_lat', 'min_lon', 'max_lon'
            resolution: H3 resolution (uses default if None)
            
        Returns:
            Set of H3 indices covering the area
        """
        if resolution is None:
            resolution = self.resolution
            
        # Create polygon from bounds
        polygon = [
            (bounds['min_lat'], bounds['min_lon']),
            (bounds['min_lat'], bounds['max_lon']),
            (bounds['max_lat'], bounds['max_lon']),
            (bounds['max_lat'], bounds['min_lon']),
            (bounds['min_lat'], bounds['min_lon'])
        ]
        
        try:
            # Convert to GeoJSON format
            geojson = {
                "type": "Polygon",
                "coordinates": [[(lon, lat) for lat, lon in polygon]]
            }
            return h3.polyfill_geojson(geojson, resolution)
        except Exception as e:
            print(f"Error creating H3 grid: {e}")
            return set()
    
    def get_hexagon_boundary(self, h3_index):
        """
        Get boundary coordinates of H3 hexagon
        
        Args:
            h3_index: H3 index string
            
        Returns:
            List of (lat, lon) tuples forming the boundary
        """
        try:
            return h3.h3_to_geo_boundary(h3_index)
        except Exception:
            return []
    
    def calculate_distance(self, h3_index1, h3_index2):
        """
        Calculate grid distance between two H3 indices
        
        Args:
            h3_index1: First H3 index
            h3_index2: Second H3 index
            
        Returns:
            Grid distance (number of hexagons)
        """
        try:
            return h3.h3_distance(h3_index1, h3_index2)
        except Exception:
            return float('inf')
    
    def build_adjacency_dict(self, h3_indices):
        """
        Build adjacency dictionary for a set of H3 indices
        
        Args:
            h3_indices: Collection of H3 indices
            
        Returns:
            Dictionary mapping each H3 to its neighbors in the set
        """
        h3_set = set(h3_indices)
        adjacency = defaultdict(set)
        
        for h3_idx in h3_set:
            neighbors = self.get_neighbors(h3_idx, ring_size=1)
            adjacency[h3_idx] = neighbors.intersection(h3_set) - {h3_idx}
            
        return dict(adjacency)
    
    def get_resolution_info(self):
        """
        Get information about current H3 resolution
        
        Returns:
            Dictionary with resolution details
        """
        # Approximate values for different resolutions
        resolution_info = {
            0: {"avg_area_km2": 4250546.85, "avg_edge_km": 1107.71},
            1: {"avg_area_km2": 607220.98, "avg_edge_km": 418.68},
            2: {"avg_area_km2": 86745.85, "avg_edge_km": 158.24},
            3: {"avg_area_km2": 12392.26, "avg_edge_km": 59.81},
            4: {"avg_area_km2": 1770.32, "avg_edge_km": 22.61},
            5: {"avg_area_km2": 252.90, "avg_edge_km": 8.54},
            6: {"avg_area_km2": 36.13, "avg_edge_km": 3.23},
            7: {"avg_area_km2": 5.16, "avg_edge_km": 1.22},
            8: {"avg_area_km2": 0.74, "avg_edge_km": 0.46},
            9: {"avg_area_km2": 0.11, "avg_edge_km": 0.17},
            10: {"avg_area_km2": 0.015, "avg_edge_km": 0.065},
        }
        
        info = resolution_info.get(self.resolution, {})
        info['resolution'] = self.resolution
        return info


def test_h3_processor():
    """Test H3 processor functionality"""
    processor = H3Processor()
    
    print("Testing H3 Processor...")
    print("-" * 50)
    
    # Test 1: Convert coordinates
    print("\n1. Converting NYC coordinates to H3...")
    nyc_lat, nyc_lon = 40.7128, -74.0060
    h3_idx = processor.lat_lon_to_h3(nyc_lat, nyc_lon)
    print(f"   NYC Center: ({nyc_lat}, {nyc_lon})")
    print(f"   H3 Index: {h3_idx}")
    
    # Test 2: Get center back
    print("\n2. Converting H3 back to coordinates...")
    center = processor.h3_to_center(h3_idx)
    print(f"   Center: {center}")
    
    # Test 3: Get neighbors
    print("\n3. Getting neighbors...")
    neighbors = processor.get_neighbors(h3_idx, ring_size=1)
    print(f"   Number of neighbors (ring 1): {len(neighbors)}")
    
    # Test 4: Resolution info
    print("\n4. Resolution info...")
    info = processor.get_resolution_info()
    print(f"   Resolution: {info['resolution']}")
    print(f"   Avg area: {info.get('avg_area_km2', 'N/A')} km²")
    print(f"   Avg edge: {info.get('avg_edge_km', 'N/A')} km")
    
    # Test 5: Sample DataFrame
    print("\n5. Processing sample DataFrame...")
    sample_df = pd.DataFrame({
        'latitude': [40.7128, 40.7580, 40.6892, 40.7484],
        'longitude': [-74.0060, -73.9855, -74.0445, -73.9857],
        'severity': [2, 5, 1, 3]
    })
    
    result_df = processor.add_h3_column(sample_df)
    print(f"   Added H3 column to {len(result_df)} records")
    print(f"   Unique hexagons: {result_df['h3_index'].nunique()}")
    
    print("\n" + "=" * 50)
    print("H3 Processor Test Complete!")
    
    return processor


if __name__ == "__main__":
    test_h3_processor()
