"""
NOVELTY 1: Context Aware Adaptive Risk Scoring (CARS)
Dynamic risk scoring where weights change based on context
"""
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CONTEXT_WINDOWS, WEATHER_THRESHOLDS


class ContextAdaptiveRiskScoring:
    """
    NOVELTY 1: Context Aware Adaptive Risk Scoring (CARS)
    
    Instead of static risk formula, CARS uses dynamic weights that
    change based on:
    - Peak hours (morning/evening)
    - Weather conditions (rain, wind)
    - Proximity to hospitals
    - Road importance
    - Time of day
    
    Formula: CARS = Σ wi(context) * features
    
    This is a NOVEL CONTRIBUTION that makes risk scoring adaptive
    to real-world conditions.
    """
    
    def __init__(self):
        self.context_windows = CONTEXT_WINDOWS
        self.weather_thresholds = WEATHER_THRESHOLDS
        
        # Base weights for features
        self.base_weights = {
            'incident_count': 0.25,
            'severity': 0.20,
            'weather_risk': 0.15,
            'historical_risk': 0.15,
            'hospital_proximity': 0.10,
            'road_importance': 0.10,
            'temporal_risk': 0.05
        }
        
        # Context-specific weight multipliers
        self.context_multipliers = {
            'morning_peak': {
                'incident_count': 1.3,
                'severity': 1.2,
                'road_importance': 1.5,
                'temporal_risk': 1.4
            },
            'evening_peak': {
                'incident_count': 1.4,
                'severity': 1.3,
                'road_importance': 1.6,
                'temporal_risk': 1.5
            },
            'night': {
                'incident_count': 0.8,
                'severity': 1.5,  # Severity matters more at night
                'hospital_proximity': 1.3,  # Response time critical
                'weather_risk': 1.2
            },
            'rainy': {
                'weather_risk': 2.0,  # Double weather importance
                'incident_count': 1.2,
                'severity': 1.1,
                'road_importance': 1.3
            },
            'normal': {
                # All multipliers = 1.0 (base weights)
            }
        }
        
    def get_context(self, hour, is_raining=False, wind_speed=0, visibility=10000):
        """
        Determine current context based on conditions
        
        Args:
            hour: Hour of day (0-23)
            is_raining: Whether it's raining
            wind_speed: Wind speed in km/h
            visibility: Visibility in meters
            
        Returns:
            Context string and context details
        """
        contexts = []
        
        # Check weather conditions first (highest priority)
        if is_raining or wind_speed > self.weather_thresholds['high_wind']:
            contexts.append('rainy')
        
        # Check time-based contexts
        morning_start, morning_end = self.context_windows['morning_peak']
        evening_start, evening_end = self.context_windows['evening_peak']
        night_start, night_end = self.context_windows['night']
        
        if morning_start <= hour < morning_end:
            contexts.append('morning_peak')
        elif evening_start <= hour < evening_end:
            contexts.append('evening_peak')
        elif hour >= night_start or hour < night_end:
            contexts.append('night')
        
        if not contexts:
            contexts.append('normal')
        
        # Primary context (first in priority order)
        primary_context = contexts[0]
        
        return primary_context, {
            'primary': primary_context,
            'all_contexts': contexts,
            'hour': hour,
            'is_raining': is_raining,
            'wind_speed': wind_speed,
            'visibility': visibility
        }
    
    def get_adaptive_weights(self, context):
        """
        Get adaptive weights for a given context
        
        Args:
            context: Context string
            
        Returns:
            Dictionary of adapted weights
        """
        weights = self.base_weights.copy()
        multipliers = self.context_multipliers.get(context, {})
        
        # Apply multipliers
        for feature, multiplier in multipliers.items():
            if feature in weights:
                weights[feature] *= multiplier
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def calculate_cars_score(self, features, context=None, hour=None, 
                             is_raining=False, wind_speed=0, visibility=10000):
        """
        Calculate Context Aware Adaptive Risk Score
        
        Args:
            features: Dictionary or Series with feature values
            context: Pre-determined context (optional)
            hour: Hour of day (used if context not provided)
            is_raining: Weather condition
            wind_speed: Wind speed
            visibility: Visibility
            
        Returns:
            CARS score (0-1) and score breakdown
        """
        # Determine context if not provided
        if context is None:
            if hour is None:
                hour = datetime.now().hour
            context, context_details = self.get_context(hour, is_raining, wind_speed, visibility)
        else:
            context_details = {'primary': context}
        
        # Get adaptive weights
        weights = self.get_adaptive_weights(context)
        
        # Extract and normalize feature values
        feature_values = self._extract_features(features)
        
        # Calculate weighted score
        score = 0
        breakdown = {}
        
        for feature, weight in weights.items():
            value = feature_values.get(feature, 0)
            contribution = weight * value
            score += contribution
            breakdown[feature] = {
                'value': value,
                'weight': weight,
                'contribution': contribution
            }
        
        # Clip to [0, 1]
        score = np.clip(score, 0, 1)
        
        return score, {
            'score': score,
            'context': context,
            'context_details': context_details,
            'weights': weights,
            'breakdown': breakdown
        }
    
    def calculate_batch_scores(self, df):
        """
        Calculate CARS scores for entire DataFrame
        
        Args:
            df: DataFrame with required features
            
        Returns:
            DataFrame with CARS scores added
        """
        df = df.copy()
        
        scores = []
        contexts = []
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Extract context parameters
            hour = row.get('hour', 12)
            is_raining = row.get('is_raining', False) or row.get('precipitation', 0) > 0
            wind_speed = row.get('wind_speed', 0)
            visibility = row.get('visibility', 10000)
            
            # Calculate score
            score, details = self.calculate_cars_score(
                row, 
                hour=hour,
                is_raining=is_raining,
                wind_speed=wind_speed,
                visibility=visibility
            )
            
            scores.append(score)
            contexts.append(details['context'])
        
        df['cars_score'] = scores
        df['cars_context'] = contexts
        
        # Add risk category
        df['cars_risk_category'] = pd.cut(
            df['cars_score'],
            bins=[-np.inf, 0.3, 0.6, 0.8, np.inf],
            labels=['low', 'medium', 'high', 'critical']
        )
        
        return df
    
    def _extract_features(self, features):
        """
        Extract and normalize feature values
        
        Args:
            features: Dictionary or Series with raw features
            
        Returns:
            Dictionary with normalized feature values (0-1)
        """
        if isinstance(features, pd.Series):
            features = features.to_dict()
        
        normalized = {}
        
        # Incident count (normalize assuming max ~100)
        incident_count = features.get('incident_count', features.get('zone_incident_count', 0))
        normalized['incident_count'] = min(incident_count / 100, 1.0) if incident_count else 0
        
        # Severity (normalize assuming max ~50)
        severity = features.get('severity', features.get('total_severity', features.get('avg_severity', 0)))
        normalized['severity'] = min(severity / 50, 1.0) if severity else 0
        
        # Weather risk (already 0-1 if calculated)
        normalized['weather_risk'] = features.get('weather_risk_score', 
                                                   features.get('rain_intensity_score', 0))
        
        # Historical risk (already 0-1)
        normalized['historical_risk'] = features.get('historical_risk', 
                                                     features.get('base_risk_score', 0))
        
        # Hospital proximity (inverse - closer = higher risk for response)
        hospital_km = features.get('nearest_hospital_km', 5)
        normalized['hospital_proximity'] = 1 - min(hospital_km / 10, 1.0)
        
        # Road importance
        normalized['road_importance'] = features.get('is_major_road', 
                                                     features.get('zone_density_score', 0))
        
        # Temporal risk
        is_peak = features.get('is_peak_hour', 0)
        is_night = features.get('is_night', 0)
        normalized['temporal_risk'] = max(is_peak * 0.8, is_night * 0.6)
        
        return normalized
    
    def explain_score(self, score_details):
        """
        Generate human-readable explanation of CARS score
        
        Args:
            score_details: Details from calculate_cars_score
            
        Returns:
            Explanation string
        """
        explanation = []
        
        explanation.append(f"CARS Score: {score_details['score']:.2f}")
        explanation.append(f"Context: {score_details['context']}")
        explanation.append("\nFactor Contributions:")
        
        # Sort by contribution
        breakdown = score_details['breakdown']
        sorted_factors = sorted(breakdown.items(), 
                               key=lambda x: x[1]['contribution'], 
                               reverse=True)
        
        for factor, details in sorted_factors:
            contribution_pct = details['contribution'] / score_details['score'] * 100 if score_details['score'] > 0 else 0
            explanation.append(
                f"  - {factor}: {details['contribution']:.3f} "
                f"(weight: {details['weight']:.2f}, value: {details['value']:.2f}, "
                f"{contribution_pct:.1f}% of total)"
            )
        
        return "\n".join(explanation)
    
    def get_weight_comparison(self):
        """
        Get comparison of weights across all contexts
        
        Returns:
            DataFrame comparing weights
        """
        contexts = ['normal', 'morning_peak', 'evening_peak', 'night', 'rainy']
        
        data = []
        for context in contexts:
            weights = self.get_adaptive_weights(context)
            weights['context'] = context
            data.append(weights)
        
        return pd.DataFrame(data).set_index('context')


def test_cars():
    """Test CARS functionality"""
    print("Testing Context Aware Adaptive Risk Scoring (CARS)...")
    print("=" * 70)
    
    cars = ContextAdaptiveRiskScoring()
    
    # Test 1: Context detection
    print("\n1. Context Detection:")
    test_cases = [
        (8, False, 10, 10000),   # Morning peak
        (17, False, 10, 10000),  # Evening peak
        (23, False, 10, 10000),  # Night
        (12, True, 10, 10000),   # Rainy midday
        (14, False, 10, 10000),  # Normal
    ]
    
    for hour, rain, wind, vis in test_cases:
        context, details = cars.get_context(hour, rain, wind, vis)
        print(f"   Hour={hour}, Rain={rain} → Context: {context}")
    
    # Test 2: Weight comparison
    print("\n2. Weight Comparison Across Contexts:")
    weight_df = cars.get_weight_comparison()
    print(weight_df.round(3))
    
    # Test 3: Score calculation
    print("\n3. Sample Score Calculations:")
    
    sample_features = {
        'incident_count': 25,
        'severity': 15,
        'weather_risk_score': 0.3,
        'historical_risk': 0.5,
        'nearest_hospital_km': 2,
        'is_major_road': 1,
        'is_peak_hour': 1
    }
    
    for context in ['normal', 'morning_peak', 'rainy']:
        score, details = cars.calculate_cars_score(sample_features, context=context)
        print(f"\n   Context: {context}")
        print(f"   CARS Score: {score:.3f}")
    
    # Test 4: Explanation
    print("\n4. Score Explanation:")
    score, details = cars.calculate_cars_score(sample_features, context='morning_peak')
    explanation = cars.explain_score(details)
    print(explanation)
    
    print("\n" + "=" * 70)
    print("CARS Test Complete!")
    
    return cars


if __name__ == "__main__":
    test_cars()
