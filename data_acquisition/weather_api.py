"""
Open-Meteo Weather API Client
Data Source: Open-Meteo (Free, No Auth Required)
Endpoint: https://api.open-meteo.com/v1/forecast
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import OPEN_METEO_API, NYC_LAT, NYC_LON, API_CALL_DELAY


class OpenMeteoAPI:
    """Client for Open-Meteo Weather API"""
    
    def __init__(self):
        self.base_url = OPEN_METEO_API
        self.session = requests.Session()
        
    def fetch_current_weather(self, latitude=NYC_LAT, longitude=NYC_LON):
        """
        Fetch current weather conditions
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            
        Returns:
            Dictionary with current weather data
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m,wind_direction_10m",
            "timezone": "America/New_York"
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return self._process_current(data)
        except requests.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return {}
    
    def fetch_hourly_forecast(self, latitude=NYC_LAT, longitude=NYC_LON, days=7):
        """
        Fetch hourly weather forecast
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            days: Number of forecast days (1-16)
            
        Returns:
            pandas DataFrame with hourly forecast
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,visibility,wind_speed_10m,wind_direction_10m",
            "forecast_days": min(days, 16),
            "timezone": "America/New_York"
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return self._process_hourly(data)
        except requests.RequestException as e:
            print(f"Error fetching weather forecast: {e}")
            return pd.DataFrame()
    
    def fetch_historical_weather(self, latitude=NYC_LAT, longitude=NYC_LON, 
                                  start_date=None, end_date=None):
        """
        Fetch historical weather data
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            pandas DataFrame with historical weather
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        # Use archive endpoint for historical data
        archive_url = "https://archive-api.open-meteo.com/v1/archive"
        
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,visibility,wind_speed_10m,wind_direction_10m",
            "timezone": "America/New_York"
        }
        
        try:
            response = self.session.get(archive_url, params=params)
            response.raise_for_status()
            data = response.json()
            return self._process_hourly(data)
        except requests.RequestException as e:
            print(f"Error fetching historical weather: {e}")
            return pd.DataFrame()
    
    def fetch_weather_for_locations(self, locations, forecast_days=1):
        """
        Fetch weather for multiple locations
        
        Args:
            locations: List of (latitude, longitude) tuples
            forecast_days: Number of forecast days
            
        Returns:
            Dictionary mapping location to weather DataFrame
        """
        results = {}
        
        for lat, lon in locations:
            key = f"{lat:.4f},{lon:.4f}"
            df = self.fetch_hourly_forecast(lat, lon, forecast_days)
            if not df.empty:
                df['latitude'] = lat
                df['longitude'] = lon
                results[key] = df
            time.sleep(API_CALL_DELAY)
            
        return results
    
    def _process_current(self, data):
        """Process current weather response"""
        if 'current' not in data:
            return {}
            
        current = data['current']
        return {
            'timestamp': datetime.now(),
            'temperature': current.get('temperature_2m'),
            'humidity': current.get('relative_humidity_2m'),
            'precipitation': current.get('precipitation', 0),
            'rain': current.get('rain', 0),
            'weather_code': current.get('weather_code'),
            'wind_speed': current.get('wind_speed_10m'),
            'wind_direction': current.get('wind_direction_10m'),
            'weather_description': self._get_weather_description(current.get('weather_code'))
        }
    
    def _process_hourly(self, data):
        """Process hourly forecast response"""
        if 'hourly' not in data:
            return pd.DataFrame()
            
        hourly = data['hourly']
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(hourly.get('time', [])),
            'temperature': hourly.get('temperature_2m', []),
            'humidity': hourly.get('relative_humidity_2m', []),
            'precipitation': hourly.get('precipitation', []),
            'rain': hourly.get('rain', []),
            'weather_code': hourly.get('weather_code', []),
            'visibility': hourly.get('visibility', []),
            'wind_speed': hourly.get('wind_speed_10m', []),
            'wind_direction': hourly.get('wind_direction_10m', [])
        })
        
        # Add weather description
        df['weather_description'] = df['weather_code'].apply(self._get_weather_description)
        
        # Add derived features
        df['is_raining'] = df['precipitation'] > 0
        df['is_heavy_rain'] = df['precipitation'] > 5.0
        df['is_windy'] = df['wind_speed'] > 30
        
        # Add fetch timestamp
        df['fetched_at'] = datetime.now()
        
        return df
    
    def _get_weather_description(self, code):
        """Convert WMO weather code to description"""
        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            66: "Light freezing rain",
            67: "Heavy freezing rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }
        return weather_codes.get(code, "Unknown")


def test_api():
    """Test the Open-Meteo Weather API"""
    api = OpenMeteoAPI()
    
    print("Testing Open-Meteo Weather API...")
    print("-" * 50)
    
    # Test 1: Current weather
    print("\n1. Fetching current weather for NYC...")
    current = api.fetch_current_weather()
    if current:
        print(f"   Temperature: {current.get('temperature')}°C")
        print(f"   Humidity: {current.get('humidity')}%")
        print(f"   Precipitation: {current.get('precipitation')} mm")
        print(f"   Wind Speed: {current.get('wind_speed')} km/h")
        print(f"   Conditions: {current.get('weather_description')}")
    
    # Test 2: Hourly forecast
    print("\n2. Fetching hourly forecast (next 24 hours)...")
    df = api.fetch_hourly_forecast(days=1)
    print(f"   Records fetched: {len(df)}")
    if not df.empty:
        print(f"   Columns: {list(df.columns)}")
        print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Test 3: Historical weather
    print("\n3. Fetching historical weather (last 7 days)...")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    df_hist = api.fetch_historical_weather(start_date=start_date, end_date=end_date)
    print(f"   Records fetched: {len(df_hist)}")
    
    print("\n" + "=" * 50)
    print("API Test Complete!")
    
    return df


if __name__ == "__main__":
    test_api()
