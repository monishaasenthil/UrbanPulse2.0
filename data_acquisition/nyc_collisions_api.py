"""
NYC Motor Vehicle Collisions API Client
Data Source: NYC Open Data
Endpoint: https://data.cityofnewyork.us/resource/h9gi-nx95.json
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import NYC_COLLISIONS_API, API_CALL_DELAY, MAX_RECORDS_PER_CALL


class NYCCollisionsAPI:
    """Client for NYC Motor Vehicle Collisions API"""
    
    def __init__(self):
        self.base_url = NYC_COLLISIONS_API
        self.session = requests.Session()
        
    def fetch_recent_collisions(self, limit=500):
        """
        Fetch most recent collision records
        
        Args:
            limit: Maximum number of records to fetch
            
        Returns:
            pandas DataFrame with collision data
        """
        params = {
            "$limit": min(limit, MAX_RECORDS_PER_CALL),
            "$order": "crash_date DESC, crash_time DESC"
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return self._process_response(data)
        except requests.RequestException as e:
            print(f"Error fetching collision data: {e}")
            return pd.DataFrame()
    
    def fetch_by_date_range(self, start_date, end_date, limit=5000):
        """
        Fetch collisions within a date range
        
        Args:
            start_date: Start date (YYYY-MM-DD format or datetime)
            end_date: End date (YYYY-MM-DD format or datetime)
            limit: Maximum records to fetch
            
        Returns:
            pandas DataFrame with collision data
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        start_str = start_date.strftime("%Y-%m-%dT00:00:00")
        end_str = end_date.strftime("%Y-%m-%dT23:59:59")
        
        all_data = []
        offset = 0
        
        while offset < limit:
            batch_limit = min(MAX_RECORDS_PER_CALL, limit - offset)
            
            params = {
                "$where": f"crash_date between '{start_str}' and '{end_str}'",
                "$limit": batch_limit,
                "$offset": offset,
                "$order": "crash_date DESC, crash_time DESC"
            }
            
            try:
                response = self.session.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                    
                all_data.extend(data)
                offset += len(data)
                
                if len(data) < batch_limit:
                    break
                    
                time.sleep(API_CALL_DELAY)
                
            except requests.RequestException as e:
                print(f"Error fetching collision data: {e}")
                break
                
        return self._process_response(all_data)
    
    def fetch_by_borough(self, borough, limit=1000):
        """
        Fetch collisions for a specific borough
        
        Args:
            borough: Borough name (MANHATTAN, BROOKLYN, QUEENS, BRONX, STATEN ISLAND)
            limit: Maximum records to fetch
            
        Returns:
            pandas DataFrame with collision data
        """
        params = {
            "$where": f"borough='{borough.upper()}'",
            "$limit": min(limit, MAX_RECORDS_PER_CALL),
            "$order": "crash_date DESC"
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return self._process_response(data)
        except requests.RequestException as e:
            print(f"Error fetching collision data: {e}")
            return pd.DataFrame()
    
    def _process_response(self, data):
        """
        Process API response into structured DataFrame
        
        Args:
            data: Raw JSON response
            
        Returns:
            Processed pandas DataFrame
        """
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # Define columns we need
        required_columns = [
            'crash_date', 'crash_time', 'latitude', 'longitude',
            'number_of_persons_injured', 'number_of_persons_killed',
            'contributing_factor_vehicle_1', 'vehicle_type_code1',
            'borough', 'zip_code', 'on_street_name', 'cross_street_name'
        ]
        
        # Add missing columns with None
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
                
        # Select and rename columns
        df = df[required_columns].copy()
        df.columns = [
            'crash_date', 'crash_time', 'latitude', 'longitude',
            'persons_injured', 'persons_killed',
            'contributing_factor', 'vehicle_type',
            'borough', 'zip_code', 'street_name', 'cross_street'
        ]
        
        # Convert data types
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['persons_injured'] = pd.to_numeric(df['persons_injured'], errors='coerce').fillna(0).astype(int)
        df['persons_killed'] = pd.to_numeric(df['persons_killed'], errors='coerce').fillna(0).astype(int)
        
        # Parse datetime
        df['crash_date'] = pd.to_datetime(df['crash_date'], errors='coerce')
        
        # Create combined datetime
        df['crash_datetime'] = df.apply(
            lambda row: self._combine_datetime(row['crash_date'], row['crash_time']),
            axis=1
        )
        
        # Add severity score
        df['severity'] = df['persons_killed'] * 5 + df['persons_injured']
        
        # Add fetch timestamp
        df['fetched_at'] = datetime.now()
        
        return df
    
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


def test_api():
    """Test the NYC Collisions API"""
    api = NYCCollisionsAPI()
    
    print("Testing NYC Collisions API...")
    print("-" * 50)
    
    # Test 1: Fetch recent collisions
    print("\n1. Fetching recent collisions (limit=10)...")
    df = api.fetch_recent_collisions(limit=10)
    print(f"   Records fetched: {len(df)}")
    if not df.empty:
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df['crash_date'].min()} to {df['crash_date'].max()}")
    
    # Test 2: Fetch by date range
    print("\n2. Fetching by date range (last 7 days)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    df = api.fetch_by_date_range(start_date, end_date, limit=100)
    print(f"   Records fetched: {len(df)}")
    
    print("\n" + "=" * 50)
    print("API Test Complete!")
    
    return df


if __name__ == "__main__":
    test_api()
