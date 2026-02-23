"""
Urban Pulse 2.0 - Configuration Settings
"""
import os

# API Configuration
NYC_COLLISIONS_API = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"
OPEN_METEO_API = "https://api.open-meteo.com/v1/forecast"

# NYC Coordinates (default center)
NYC_LAT = 40.7128
NYC_LON = -74.0060

# H3 Configuration
H3_RESOLUTION = 8  # Micro-zone resolution

# Data Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BRONZE_DIR = os.path.join(DATA_DIR, "bronze")
SILVER_DIR = os.path.join(DATA_DIR, "silver")
GOLD_DIR = os.path.join(DATA_DIR, "gold")
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Model Configuration
RISK_THRESHOLD_LOW = 0.3
RISK_THRESHOLD_MEDIUM = 0.6
RISK_THRESHOLD_HIGH = 0.8

# Context Windows
CONTEXT_WINDOWS = {
    "morning_peak": (7, 10),    # 7 AM - 10 AM
    "evening_peak": (16, 19),   # 4 PM - 7 PM
    "night": (22, 6),           # 10 PM - 6 AM
    "normal": None              # Default
}

# Weather Thresholds
WEATHER_THRESHOLDS = {
    "heavy_rain": 5.0,      # mm/hour
    "light_rain": 0.5,      # mm/hour
    "high_wind": 30.0,      # km/h
    "low_visibility": 1000  # meters
}

# Risk Propagation Parameters
PROPAGATION_DECAY = 0.5  # How much risk decays to neighbors
MAX_PROPAGATION_HOPS = 2  # Maximum neighbor distance

# Feedback Loop Parameters
FEEDBACK_LEARNING_RATE = 0.01
FEEDBACK_WINDOW_HOURS = 24

# API Rate Limits
API_CALL_DELAY = 1  # seconds between calls
MAX_RECORDS_PER_CALL = 1000
