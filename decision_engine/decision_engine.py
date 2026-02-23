"""
Decision Engine - Unified Decision Making System
Orchestrates all decision components based on risk analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import RISK_THRESHOLD_LOW, RISK_THRESHOLD_MEDIUM, RISK_THRESHOLD_HIGH, OUTPUTS_DIR
from decision_engine.signal_controller import SignalController
from decision_engine.emergency_router import EmergencyRouter
from decision_engine.alert_generator import AlertGenerator


class DecisionEngine:
    """
    Unified Decision Engine for Urban Pulse
    
    Integrates:
    - Signal Controller (traffic signal tuning)
    - Emergency Router (priority vehicle routing)
    - Alert Generator (relief center alerts)
    
    Makes decisions based on:
    - CARS scores (Novelty 1)
    - Propagated risk (Novelty 2)
    - Feedback learning (Novelty 3)
    - Explainability (Novelty 4)
    """
    
    def __init__(self):
        self.signal_controller = SignalController()
        self.emergency_router = EmergencyRouter()
        self.alert_generator = AlertGenerator()
        
        self.decision_history = []
        self.thresholds = {
            'low': RISK_THRESHOLD_LOW,
            'medium': RISK_THRESHOLD_MEDIUM,
            'high': RISK_THRESHOLD_HIGH
        }
        
    def process_risk_data(self, zone_data, weather_data=None):
        """
        Process risk data and generate all decisions
        
        Args:
            zone_data: DataFrame with zone risk scores
            weather_data: Optional weather conditions
            
        Returns:
            Dictionary with all decisions and outputs
        """
        print("\n" + "=" * 70)
        print("URBAN PULSE - DECISION ENGINE")
        print("=" * 70)
        
        results = {
            'timestamp': datetime.now(),
            'zones_processed': len(zone_data),
            'decisions': []
        }
        
        # 1. Generate Signal Plans
        print("\n[1/4] Generating Signal Tuning Plans...")
        signal_plan = self.signal_controller.generate_signal_plan(zone_data)
        results['signal_plan'] = signal_plan
        results['signal_summary'] = self.signal_controller.get_summary(signal_plan)
        print(f"      Generated plans for {len(signal_plan)} zones")
        
        # 2. Build Routing Graph
        print("\n[2/4] Building Emergency Routing Graph...")
        self.emergency_router.build_routing_graph(zone_data)
        results['routing_graph_built'] = True
        print(f"      Graph ready with {len(zone_data)} zones")
        
        # 3. Generate Alerts
        print("\n[3/4] Generating Alerts...")
        alerts = self.alert_generator.generate_alerts(zone_data, weather_data)
        relief_alerts = self.alert_generator.generate_relief_center_alerts(zone_data, weather_data)
        results['alerts'] = alerts
        results['relief_alerts'] = relief_alerts
        results['alert_summary'] = self.alert_generator.get_alert_summary()
        print(f"      Generated {len(alerts)} alerts")
        
        # 4. Generate Priority Decisions
        print("\n[4/4] Generating Priority Decisions...")
        priority_decisions = self._generate_priority_decisions(zone_data, weather_data)
        results['priority_decisions'] = priority_decisions
        print(f"      Generated {len(priority_decisions)} priority decisions")
        
        # Record in history
        self.decision_history.append(results)
        
        print("\n" + "-" * 70)
        print("DECISION ENGINE COMPLETE")
        print("-" * 70)
        
        return results
    
    def _generate_priority_decisions(self, zone_data, weather_data=None):
        """
        Generate priority decisions for high-risk zones
        
        Args:
            zone_data: DataFrame with zone data
            weather_data: Weather conditions
            
        Returns:
            List of priority decisions
        """
        decisions = []
        
        # Get risk column
        risk_col = 'propagated_risk' if 'propagated_risk' in zone_data.columns else 'base_risk_score'
        
        for _, zone in zone_data.iterrows():
            risk_score = zone.get(risk_col, 0)
            h3_index = zone.get('h3_index', 'unknown')
            
            if risk_score > self.thresholds['high']:
                # Critical risk - multiple actions
                decisions.append({
                    'zone': h3_index,
                    'risk_score': risk_score,
                    'priority': 'critical',
                    'actions': [
                        {'type': 'signal_change', 'description': 'Extend green phase'},
                        {'type': 'alert', 'description': 'Notify emergency services'},
                        {'type': 'routing', 'description': 'Prepare alternative routes'}
                    ],
                    'timestamp': datetime.now()
                })
            elif risk_score > self.thresholds['medium']:
                # High risk - preventive actions
                decisions.append({
                    'zone': h3_index,
                    'risk_score': risk_score,
                    'priority': 'high',
                    'actions': [
                        {'type': 'signal_change', 'description': 'Adjust signal timing'},
                        {'type': 'monitoring', 'description': 'Increase monitoring'}
                    ],
                    'timestamp': datetime.now()
                })
        
        return decisions
    
    def get_routing_for_incident(self, incident_location, destination_type='hospital'):
        """
        Get emergency routing for an incident
        
        Args:
            incident_location: H3 index or (lat, lon)
            destination_type: Type of destination
            
        Returns:
            Routing directive
        """
        return self.emergency_router.generate_routing_directive(
            incident_location, destination_type
        )
    
    def export_all_outputs(self, output_dir=None):
        """
        Export all decision outputs to files
        
        Args:
            output_dir: Output directory
            
        Returns:
            Dictionary with file paths
        """
        if output_dir is None:
            output_dir = OUTPUTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        paths = {}
        
        # Export signal plans
        if self.decision_history:
            latest = self.decision_history[-1]
            
            if 'signal_plan' in latest and not latest['signal_plan'].empty:
                path = os.path.join(output_dir, f'signal_tuning_plan_{timestamp}.csv')
                latest['signal_plan'].to_csv(path, index=False)
                paths['signal_plan'] = path
            
            if 'relief_alerts' in latest and not latest['relief_alerts'].empty:
                path = os.path.join(output_dir, f'relief_center_alerts_{timestamp}.csv')
                latest['relief_alerts'].to_csv(path, index=False)
                paths['relief_alerts'] = path
            
            if 'priority_decisions' in latest:
                path = os.path.join(output_dir, f'priority_decisions_{timestamp}.csv')
                pd.DataFrame(latest['priority_decisions']).to_csv(path, index=False)
                paths['priority_decisions'] = path
        
        print(f"Exported {len(paths)} output files to {output_dir}")
        return paths
    
    def get_dashboard_data(self):
        """
        Get data formatted for dashboard display
        
        Returns:
            Dictionary with dashboard-ready data
        """
        if not self.decision_history:
            return {}
        
        latest = self.decision_history[-1]
        
        return {
            'timestamp': latest['timestamp'].isoformat(),
            'zones_processed': latest['zones_processed'],
            'signal_summary': latest.get('signal_summary', {}),
            'alert_summary': latest.get('alert_summary', {}),
            'priority_decisions_count': len(latest.get('priority_decisions', [])),
            'critical_zones': len([d for d in latest.get('priority_decisions', []) 
                                  if d.get('priority') == 'critical'])
        }


def test_decision_engine():
    """Test decision engine"""
    print("Testing Decision Engine...")
    
    # Create sample zone data
    np.random.seed(42)
    n_zones = 20
    
    from data_engineering.h3_processor import H3Processor
    h3_proc = H3Processor()
    
    # Generate H3 indices around NYC
    center_h3 = h3_proc.lat_lon_to_h3(40.7128, -74.0060)
    h3_indices = list(h3_proc.get_neighbors(center_h3, ring_size=2))[:n_zones]
    
    zone_data = pd.DataFrame({
        'h3_index': h3_indices,
        'base_risk_score': np.random.uniform(0.2, 0.9, n_zones),
        'propagated_risk': np.random.uniform(0.3, 0.95, n_zones),
        'incident_count': np.random.randint(0, 30, n_zones),
        'center_lat': [h3_proc.h3_to_center(h)[0] for h in h3_indices],
        'center_lon': [h3_proc.h3_to_center(h)[1] for h in h3_indices]
    })
    
    weather_data = {
        'precipitation': 3.5,
        'wind_speed': 25,
        'visibility': 5000,
        'temperature': 15
    }
    
    # Run decision engine
    engine = DecisionEngine()
    results = engine.process_risk_data(zone_data, weather_data)
    
    # Display results
    print("\n" + "=" * 70)
    print("DECISION ENGINE RESULTS")
    print("=" * 70)
    
    print(f"\nSignal Summary: {results['signal_summary']}")
    print(f"Alert Summary: {results['alert_summary']}")
    print(f"Priority Decisions: {len(results['priority_decisions'])}")
    
    # Export outputs
    paths = engine.export_all_outputs()
    print(f"\nExported files: {list(paths.keys())}")
    
    return engine


if __name__ == "__main__":
    test_decision_engine()
