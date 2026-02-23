"""
Signal Controller - Adaptive Traffic Signal Tuning
Generates signal timing adjustments based on risk scores
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import RISK_THRESHOLD_HIGH, OUTPUTS_DIR


class SignalController:
    """
    Adaptive Signal Tuning Controller
    
    Generates signal timing plans based on:
    - Risk scores per zone
    - Traffic patterns
    - Time of day
    - Weather conditions
    """
    
    def __init__(self):
        self.base_green_time = 30  # seconds
        self.base_cycle_time = 90  # seconds
        self.min_green_time = 15
        self.max_green_time = 60
        self.signal_plans = []
        
    def generate_signal_plan(self, zone_data, risk_col='propagated_risk'):
        """
        Generate signal tuning plan for zones
        
        Args:
            zone_data: DataFrame with zone risk scores
            risk_col: Column name for risk score
            
        Returns:
            DataFrame with signal tuning recommendations
        """
        if zone_data.empty:
            return pd.DataFrame()
        
        plans = []
        
        for _, zone in zone_data.iterrows():
            risk_score = zone.get(risk_col, zone.get('base_risk_score', 0))
            h3_index = zone.get('h3_index', 'unknown')
            
            # Calculate signal adjustments
            adjustment = self._calculate_adjustment(risk_score, zone)
            
            # Convert numpy types to Python native types for JSON serialization
            plan = {
                'h3_index': str(h3_index),
                'risk_score': float(risk_score) if not pd.isna(risk_score) else 0.0,
                'risk_category': self._get_risk_category(risk_score),
                'recommended_green_time': int(adjustment['green_time']),
                'recommended_cycle_time': int(adjustment['cycle_time']),
                'green_time_change': int(adjustment['green_time'] - self.base_green_time),
                'priority_direction': str(adjustment['priority_direction']),
                'pedestrian_phase': str(adjustment['pedestrian_phase']),
                'action_required': bool(risk_score > RISK_THRESHOLD_HIGH),
                'timestamp': datetime.now()
            }
            
            plans.append(plan)
        
        plan_df = pd.DataFrame(plans)
        self.signal_plans.append(plan_df)
        
        return plan_df
    
    def _calculate_adjustment(self, risk_score, zone_data):
        """
        Calculate signal timing adjustment based on risk
        
        Args:
            risk_score: Zone risk score (0-1)
            zone_data: Zone data row
            
        Returns:
            Dictionary with timing adjustments
        """
        # Higher risk = longer green for main flow to clear traffic
        risk_factor = 1 + (risk_score * 0.5)  # Up to 50% increase
        
        green_time = int(self.base_green_time * risk_factor)
        green_time = max(self.min_green_time, min(self.max_green_time, green_time))
        
        # Cycle time adjustment
        cycle_time = int(self.base_cycle_time * (1 + risk_score * 0.2))
        
        # Determine priority direction based on context
        is_peak = zone_data.get('is_peak_hour', 0) or zone_data.get('peak_hour_ratio', 0) > 0.5
        hour = zone_data.get('hour', datetime.now().hour)
        
        if 7 <= hour < 10:
            priority_direction = 'inbound'  # Morning commute
        elif 16 <= hour < 19:
            priority_direction = 'outbound'  # Evening commute
        else:
            priority_direction = 'balanced'
        
        # Pedestrian phase adjustment
        if risk_score > 0.7:
            pedestrian_phase = 'extended'  # More time for pedestrians to clear
        elif risk_score > 0.4:
            pedestrian_phase = 'standard'
        else:
            pedestrian_phase = 'normal'
        
        return {
            'green_time': green_time,
            'cycle_time': cycle_time,
            'priority_direction': priority_direction,
            'pedestrian_phase': pedestrian_phase
        }
    
    def _get_risk_category(self, risk_score):
        """Categorize risk score"""
        if risk_score > 0.8:
            return 'critical'
        elif risk_score > 0.6:
            return 'high'
        elif risk_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def get_high_priority_signals(self, plan_df, threshold=0.6):
        """
        Get signals requiring immediate attention
        
        Args:
            plan_df: Signal plan DataFrame
            threshold: Risk threshold
            
        Returns:
            Filtered DataFrame
        """
        return plan_df[plan_df['risk_score'] > threshold].sort_values(
            'risk_score', ascending=False
        )
    
    def export_signal_plan(self, plan_df, filepath=None):
        """
        Export signal plan to CSV
        
        Args:
            plan_df: Signal plan DataFrame
            filepath: Output path
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(OUTPUTS_DIR, f'signal_tuning_plan_{timestamp}.csv')
        
        plan_df.to_csv(filepath, index=False)
        print(f"Signal plan exported to {filepath}")
        return filepath
    
    def get_summary(self, plan_df):
        """Get summary of signal plan"""
        return {
            'total_zones': len(plan_df),
            'critical_zones': len(plan_df[plan_df['risk_category'] == 'critical']),
            'high_risk_zones': len(plan_df[plan_df['risk_category'] == 'high']),
            'avg_green_time_change': plan_df['green_time_change'].mean(),
            'zones_requiring_action': plan_df['action_required'].sum()
        }
