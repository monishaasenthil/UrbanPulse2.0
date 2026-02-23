"""
Alert Generator - Weather and Risk Based Alerts
Generates alerts for relief centers and emergency services
"""
import pandas as pd
import numpy as np
from datetime import datetime
from enum import Enum
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import OUTPUTS_DIR, WEATHER_THRESHOLDS


class AlertLevel(Enum):
    INFO = 1
    WARNING = 2
    SEVERE = 3
    CRITICAL = 4


class AlertGenerator:
    """
    Alert Generation System
    
    Generates alerts based on:
    - Risk scores
    - Weather conditions
    - Incident patterns
    - Time-sensitive factors
    """
    
    def __init__(self):
        self.alerts = []
        self.alert_rules = self._initialize_rules()
        
    def _initialize_rules(self):
        """Initialize alert rules"""
        return {
            'high_risk': {
                'condition': lambda d: d.get('risk_score', 0) > 0.15,
                'level': AlertLevel.CRITICAL,
                'message': 'Critical risk level detected',
                'action': 'Immediate response required'
            },
            'elevated_risk': {
                'condition': lambda d: 0.08 < d.get('risk_score', 0) <= 0.15,
                'level': AlertLevel.SEVERE,
                'message': 'Elevated risk level',
                'action': 'Increase monitoring and prepare response'
            },
            'heavy_rain': {
                'condition': lambda d: d.get('precipitation', 0) > WEATHER_THRESHOLDS['heavy_rain'],
                'level': AlertLevel.WARNING,
                'message': 'Heavy rainfall detected',
                'action': 'Activate weather response protocols'
            },
            'low_visibility': {
                'condition': lambda d: d.get('visibility', 10000) < WEATHER_THRESHOLDS['low_visibility'],
                'level': AlertLevel.WARNING,
                'message': 'Low visibility conditions',
                'action': 'Issue travel advisories'
            },
            'high_wind': {
                'condition': lambda d: d.get('wind_speed', 0) > WEATHER_THRESHOLDS['high_wind'],
                'level': AlertLevel.WARNING,
                'message': 'High wind conditions',
                'action': 'Monitor for debris and structural hazards'
            },
            'incident_cluster': {
                'condition': lambda d: d.get('incident_count', 0) > 10,
                'level': AlertLevel.SEVERE,
                'message': 'Incident cluster detected',
                'action': 'Deploy additional resources to area'
            },
            'hospital_proximity': {
                'condition': lambda d: d.get('risk_score', 0) > 0.7 and d.get('near_hospital', 0),
                'level': AlertLevel.CRITICAL,
                'message': 'High risk near hospital zone',
                'action': 'Ensure emergency access routes are clear'
            }
        }
    
    def generate_alerts(self, zone_data, weather_data=None, risk_col='base_risk_score'):
        """
        Generate alerts based on zone and weather data
        
        Args:
            zone_data: DataFrame with zone risk information
            weather_data: Optional weather data dictionary
            risk_col: Column name for risk score
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        for _, zone in zone_data.iterrows():
            zone_dict = zone.to_dict()
            # Normalize risk_score key for rules
            zone_dict['risk_score'] = zone_dict.get(risk_col, zone_dict.get('base_risk_score', 0))
            
            # Merge weather data if available
            if weather_data:
                zone_dict.update(weather_data)
            
            # Check each rule
            for rule_name, rule in self.alert_rules.items():
                if rule['condition'](zone_dict):
                    alert = self._create_alert(
                        rule_name=rule_name,
                        zone_data=zone_dict,
                        level=rule['level'],
                        message=rule['message'],
                        action=rule['action']
                    )
                    alerts.append(alert)
        
        # Sort by severity
        alerts.sort(key=lambda x: x['level_value'], reverse=True)
        
        self.alerts.extend(alerts)
        return alerts
    
    def _create_alert(self, rule_name, zone_data, level, message, action):
        """Create an alert dictionary"""
        return {
            'alert_id': f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}_{rule_name}",
            'rule': rule_name,
            'level': level.name,
            'level_value': level.value,
            'message': message,
            'action': action,
            'zone_id': zone_data.get('h3_index', 'unknown'),
            'risk_score': zone_data.get('risk_score', zone_data.get('propagated_risk', 0)),
            'details': {
                'precipitation': zone_data.get('precipitation'),
                'wind_speed': zone_data.get('wind_speed'),
                'visibility': zone_data.get('visibility'),
                'incident_count': zone_data.get('incident_count')
            },
            'timestamp': datetime.now(),
            'status': 'active',
            'acknowledged': False
        }
    
    def generate_relief_center_alerts(self, zone_data, weather_data=None):
        """
        Generate specific alerts for relief centers
        
        Args:
            zone_data: DataFrame with zone information
            weather_data: Weather conditions
            
        Returns:
            DataFrame with relief center alerts
        """
        alerts = []
        
        # Check for severe weather
        if weather_data:
            if weather_data.get('precipitation', 0) > 5:
                alerts.append({
                    'type': 'weather',
                    'severity': 'high',
                    'message': 'Heavy precipitation - prepare for potential flooding',
                    'action': 'Open emergency shelters',
                    'resources_needed': ['sandbags', 'pumps', 'emergency_supplies']
                })
            
            if weather_data.get('wind_speed', 0) > 50:
                alerts.append({
                    'type': 'weather',
                    'severity': 'high',
                    'message': 'Severe wind conditions',
                    'action': 'Activate wind damage response',
                    'resources_needed': ['tarps', 'generators', 'first_aid']
                })
        
        # Check for high-risk zones
        if not zone_data.empty:
            critical_zones = zone_data[
                zone_data.get('risk_category', pd.Series(['low'] * len(zone_data))).isin(['critical', 'high'])
            ]
            
            if len(critical_zones) > 5:
                alerts.append({
                    'type': 'risk',
                    'severity': 'critical',
                    'message': f'{len(critical_zones)} zones at critical/high risk',
                    'action': 'Mobilize emergency response teams',
                    'resources_needed': ['ambulances', 'police', 'traffic_control'],
                    'affected_zones': critical_zones['h3_index'].tolist() if 'h3_index' in critical_zones.columns else []
                })
        
        # Add timestamp
        for alert in alerts:
            alert['timestamp'] = datetime.now()
            alert['status'] = 'pending'
        
        return pd.DataFrame(alerts)
    
    def get_active_alerts(self, level=None):
        """
        Get currently active alerts
        
        Args:
            level: Filter by alert level
            
        Returns:
            List of active alerts
        """
        active = [a for a in self.alerts if a['status'] == 'active']
        
        if level:
            active = [a for a in active if a['level'] == level.name]
        
        return active
    
    def acknowledge_alert(self, alert_id):
        """Mark an alert as acknowledged"""
        for alert in self.alerts:
            if alert['alert_id'] == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_at'] = datetime.now()
                return True
        return False
    
    def resolve_alert(self, alert_id, resolution_notes=''):
        """Mark an alert as resolved"""
        for alert in self.alerts:
            if alert['alert_id'] == alert_id:
                alert['status'] = 'resolved'
                alert['resolved_at'] = datetime.now()
                alert['resolution_notes'] = resolution_notes
                return True
        return False
    
    def export_alerts(self, filepath=None):
        """
        Export alerts to CSV
        
        Args:
            filepath: Output path
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(OUTPUTS_DIR, f'relief_center_alerts_{timestamp}.csv')
        
        if not self.alerts:
            print("No alerts to export")
            return None
        
        # Flatten alerts for CSV
        flat_alerts = []
        for alert in self.alerts:
            flat = {k: v for k, v in alert.items() if k != 'details'}
            if 'details' in alert:
                for dk, dv in alert['details'].items():
                    flat[f'detail_{dk}'] = dv
            flat_alerts.append(flat)
        
        df = pd.DataFrame(flat_alerts)
        df.to_csv(filepath, index=False)
        print(f"Alerts exported to {filepath}")
        return filepath
    
    def get_alert_summary(self):
        """Get summary of all alerts"""
        if not self.alerts:
            return {'total': 0}
        
        return {
            'total': len(self.alerts),
            'active': len([a for a in self.alerts if a['status'] == 'active']),
            'resolved': len([a for a in self.alerts if a['status'] == 'resolved']),
            'by_level': {
                'critical': len([a for a in self.alerts if a['level'] == 'CRITICAL']),
                'severe': len([a for a in self.alerts if a['level'] == 'SEVERE']),
                'warning': len([a for a in self.alerts if a['level'] == 'WARNING']),
                'info': len([a for a in self.alerts if a['level'] == 'INFO'])
            },
            'acknowledged': len([a for a in self.alerts if a.get('acknowledged', False)])
        }
