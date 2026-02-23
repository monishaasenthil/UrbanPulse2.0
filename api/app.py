"""
Urban Pulse API - Flask Backend
RESTful API for the Urban Pulse web platform
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    CORS(app)
    
    # Store for runtime data
    app.config['DATA'] = {
        'gold_data': None,
        'risk_data': None,
        'alerts': [],
        'decisions': [],
        'last_update': None
    }
    
    # ==================== ROOT ROUTE ====================
    @app.route('/', methods=['GET'])
    def index():
        """Root endpoint - API info"""
        return jsonify({
            'name': 'Urban Pulse 2.0 API',
            'version': '2.0.0',
            'status': 'running',
            'endpoints': {
                'health': '/api/health',
                'dashboard': '/api/dashboard/summary',
                'risk_zones': '/api/risk/zones',
                'alerts': '/api/alerts',
                'signals': '/api/decisions/signals'
            }
        })
    
    # ==================== HEALTH CHECK ====================
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0'
        })
    
    # ==================== DATA ENDPOINTS ====================
    @app.route('/api/data/refresh', methods=['POST'])
    def refresh_data():
        """Trigger data refresh from APIs"""
        try:
            from data_acquisition.data_fetcher import DataFetcher
            from data_engineering.pipeline import DataPipeline
            
            days_back = request.json.get('days_back', 7) if request.json else 7
            
            pipeline = DataPipeline()
            results = pipeline.run_full_pipeline(days_back=days_back, collision_limit=2000)
            
            app.config['DATA']['gold_data'] = results.get('gold_microzone')
            app.config['DATA']['last_update'] = datetime.now()
            
            return jsonify({
                'status': 'success',
                'records_processed': len(results.get('gold_microzone', [])),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/data/status', methods=['GET'])
    def data_status():
        """Get current data status"""
        gold_data = app.config['DATA']['gold_data']
        return jsonify({
            'has_data': gold_data is not None and not gold_data.empty if isinstance(gold_data, pd.DataFrame) else False,
            'record_count': len(gold_data) if gold_data is not None and isinstance(gold_data, pd.DataFrame) else 0,
            'last_update': app.config['DATA']['last_update'].isoformat() if app.config['DATA']['last_update'] else None
        })
    
    # ==================== RISK ENDPOINTS ====================
    @app.route('/api/risk/zones', methods=['GET'])
    def get_risk_zones():
        """Get all zones with risk scores"""
        try:
            gold_data = app.config['DATA']['gold_data']
            
            if gold_data is None or gold_data.empty:
                # Return sample data for demo
                gold_data = _generate_sample_data()
                app.config['DATA']['gold_data'] = gold_data
            
            # Convert to JSON-serializable format
            zones = []
            for _, row in gold_data.iterrows():
                zones.append({
                    'h3_index': row.get('h3_index', ''),
                    'lat': row.get('center_lat', 0),
                    'lon': row.get('center_lon', 0),
                    'risk_score': float(row.get('base_risk_score', 0)),
                    'propagated_risk': float(row.get('propagated_risk', row.get('base_risk_score', 0))),
                    'incident_count': int(row.get('incident_count', 0)),
                    'risk_category': row.get('risk_category', 'low')
                })
            
            return jsonify({
                'zones': zones,
                'total': len(zones),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/risk/zone/<h3_index>', methods=['GET'])
    def get_zone_detail(h3_index):
        """Get detailed risk info for a specific zone"""
        try:
            gold_data = app.config['DATA']['gold_data']
            
            if gold_data is None:
                return jsonify({'status': 'error', 'message': 'No data available'}), 404
            
            zone = gold_data[gold_data['h3_index'] == h3_index]
            
            if zone.empty:
                return jsonify({'status': 'error', 'message': 'Zone not found'}), 404
            
            row = zone.iloc[0]
            return jsonify({
                'h3_index': h3_index,
                'risk_score': float(row.get('base_risk_score', 0)),
                'propagated_risk': float(row.get('propagated_risk', 0)),
                'incident_count': int(row.get('incident_count', 0)),
                'total_severity': int(row.get('total_severity', 0)),
                'total_injured': int(row.get('total_injured', 0)),
                'risk_category': row.get('risk_category', 'low'),
                'lat': float(row.get('center_lat', 0)),
                'lon': float(row.get('center_lon', 0))
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/risk/calculate', methods=['POST'])
    def calculate_risk():
        """Calculate CARS score for given features"""
        try:
            from novelties.cars import ContextAdaptiveRiskScoring
            
            data = request.json
            cars = ContextAdaptiveRiskScoring()
            
            score, details = cars.calculate_cars_score(
                features=data.get('features', {}),
                hour=data.get('hour'),
                is_raining=data.get('is_raining', False),
                wind_speed=data.get('wind_speed', 0),
                visibility=data.get('visibility', 10000)
            )
            
            return jsonify({
                'cars_score': score,
                'context': details['context'],
                'breakdown': {k: {'contribution': v['contribution']} 
                             for k, v in details['breakdown'].items()}
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/risk/propagate', methods=['POST'])
    def propagate_risk():
        """Run risk propagation on zones"""
        try:
            from novelties.risk_propagation import PriorityAwareRiskPropagation
            
            gold_data = app.config['DATA']['gold_data']
            if gold_data is None:
                return jsonify({'status': 'error', 'message': 'No data available'}), 400
            
            propagator = PriorityAwareRiskPropagation()
            result_df = propagator.calculate_propagated_scores(gold_data)
            
            app.config['DATA']['gold_data'] = result_df
            
            summary = propagator.get_propagation_summary()
            
            return jsonify({
                'status': 'success',
                'summary': summary
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    # ==================== DECISION ENDPOINTS ====================
    @app.route('/api/decisions/generate', methods=['POST'])
    def generate_decisions():
        """Generate decisions based on current risk data"""
        try:
            from decision_engine.decision_engine import DecisionEngine
            
            gold_data = app.config['DATA']['gold_data']
            if gold_data is None:
                gold_data = _generate_sample_data()
                app.config['DATA']['gold_data'] = gold_data
            
            weather_data = request.json.get('weather') if request.json else None
            
            engine = DecisionEngine()
            results = engine.process_risk_data(gold_data, weather_data)
            
            app.config['DATA']['decisions'] = results.get('priority_decisions', [])
            app.config['DATA']['alerts'] = results.get('alerts', [])
            
            return jsonify(_convert_to_serializable({
                'status': 'success',
                'signal_summary': results.get('signal_summary', {}),
                'alert_summary': results.get('alert_summary', {}),
                'priority_decisions': len(results.get('priority_decisions', []))
            }))
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/decisions/signals', methods=['GET'])
    def get_signal_plans():
        """Get current signal tuning plans"""
        try:
            from decision_engine.signal_controller import SignalController
            
            gold_data = app.config['DATA']['gold_data']
            if gold_data is None:
                gold_data = _generate_sample_data()
                app.config['DATA']['gold_data'] = gold_data
            
            if gold_data is None or gold_data.empty:
                return jsonify({'plans': [], 'total': 0, 'summary': {}})
            
            # Determine risk column
            risk_col = 'propagated_risk' if 'propagated_risk' in gold_data.columns else 'base_risk_score'
            
            controller = SignalController()
            plan_df = controller.generate_signal_plan(gold_data, risk_col=risk_col)
            
            if plan_df.empty:
                return jsonify({'plans': [], 'total': 0, 'summary': {}})
            
            plans = plan_df.to_dict('records')
            for p in plans:
                if 'timestamp' in p:
                    p['timestamp'] = p['timestamp'].isoformat() if hasattr(p['timestamp'], 'isoformat') else str(p['timestamp'])
                # Convert numpy and pandas types to Python types
                for key, val in p.items():
                    if isinstance(val, (np.integer, np.floating, np.int64, np.float64)):
                        p[key] = float(val) if isinstance(val, (np.floating, np.float64)) else int(val)
                    elif pd.isna(val):
                        p[key] = 0
                    elif hasattr(val, 'item'):  # Handle numpy scalars
                        p[key] = val.item()
            
            return jsonify(_convert_to_serializable({
                'plans': plans,
                'total': len(plans),
                'summary': controller.get_summary(plan_df)
            }))
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/decisions/apply-signal', methods=['POST'])
    def apply_signal_changes():
        """Apply signal timing changes to a zone"""
        try:
            data = request.json or {}
            zone_id = data.get('zone_id')
            green_time = data.get('green_time')
            cycle_time = data.get('cycle_time')
            priority_direction = data.get('priority_direction')
            
            # Log the applied changes (in production, this would update actual traffic signals)
            print(f"Applied signal changes to {zone_id}: green={green_time}s, cycle={cycle_time}s, direction={priority_direction}")
            
            return jsonify({
                'status': 'success',
                'message': f'Signal changes applied to zone {zone_id}',
                'applied': {
                    'zone_id': zone_id,
                    'green_time': green_time,
                    'cycle_time': cycle_time,
                    'priority_direction': priority_direction,
                    'timestamp': datetime.now().isoformat()
                }
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/decisions/routing', methods=['POST'])
    def get_routing():
        """Get emergency routing for an incident"""
        try:
            from decision_engine.emergency_router import EmergencyRouter
            
            data = request.json or {}
            incident_location = data.get('location', '')
            destination_type = data.get('destination_type', 'hospital')
            
            gold_data = app.config['DATA']['gold_data']
            if gold_data is None:
                gold_data = _generate_sample_data()
                app.config['DATA']['gold_data'] = gold_data
            
            if gold_data is None or gold_data.empty:
                return jsonify({'status': 'error', 'message': 'No data available'}), 400
            
            # Determine risk column
            risk_col = 'propagated_risk' if 'propagated_risk' in gold_data.columns else 'base_risk_score'
            
            router = EmergencyRouter()
            router.build_routing_graph(gold_data, risk_col=risk_col)
            
            # If no location provided, use first zone as demo
            if not incident_location and 'h3_index' in gold_data.columns:
                incident_location = gold_data['h3_index'].iloc[0]
            
            directive = router.generate_routing_directive(incident_location, destination_type)
            
            # Convert datetime objects
            if 'timestamp' in directive:
                directive['timestamp'] = directive['timestamp'].isoformat()
            
            return jsonify(directive)
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    # ==================== ALERT ENDPOINTS ====================
    @app.route('/api/alerts', methods=['GET'])
    def get_alerts():
        """Get all active alerts"""
        # Always regenerate alerts to get fresh data
        gold_data = app.config['DATA']['gold_data']
        if gold_data is None:
            gold_data = _generate_sample_data()
            app.config['DATA']['gold_data'] = gold_data
        
        alerts = []
        if gold_data is not None and not gold_data.empty:
            from decision_engine.alert_generator import AlertGenerator
            
            risk_col = 'propagated_risk' if 'propagated_risk' in gold_data.columns else 'base_risk_score'
            generator = AlertGenerator()
            alerts = generator.generate_alerts(gold_data, risk_col=risk_col)
            app.config['DATA']['alerts'] = alerts
        
        # Convert timestamps and numpy types
        result_alerts = []
        for alert in alerts:
            alert_copy = dict(alert)
            if 'timestamp' in alert_copy and hasattr(alert_copy['timestamp'], 'isoformat'):
                alert_copy['timestamp'] = alert_copy['timestamp'].isoformat()
            # Convert numpy types
            for key, val in alert_copy.items():
                if isinstance(val, (np.integer, np.floating)):
                    alert_copy[key] = float(val) if isinstance(val, np.floating) else int(val)
                elif pd.isna(val):
                    alert_copy[key] = None
            result_alerts.append(alert_copy)
        
        return jsonify({
            'alerts': result_alerts,
            'total': len(result_alerts)
        })
    
    @app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
    def acknowledge_alert(alert_id):
        """Acknowledge an alert"""
        alerts = app.config['DATA'].get('alerts', [])
        
        for alert in alerts:
            if alert.get('alert_id') == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_at'] = datetime.now().isoformat()
                return jsonify({'status': 'success'})
        
        return jsonify({'status': 'error', 'message': 'Alert not found'}), 404
    
    # ==================== EXPLAINABILITY ENDPOINTS ====================
    @app.route('/api/explain/decision', methods=['POST'])
    def explain_decision():
        """Get explanation for a decision"""
        try:
            from novelties.explainability import HumanInTheLoopExplainability
            
            data = request.json or {}
            decision_type = data.get('decision_type', 'signal_change')
            risk_score = data.get('risk_score', 0.5)
            action_taken = data.get('action_taken', 'No action specified')
            features = data.get('features', {
                'hour': datetime.now().hour,
                'incident_count': 25,
                'severity': 15,
                'historical_risk': 0.6,
                'temperature': 20,
                'wind_speed': 15,
                'is_raining': False
            })
            
            try:
                explainer = HumanInTheLoopExplainability()
                explanation = explainer.explain_decision(
                    decision_type=decision_type,
                    risk_score=risk_score,
                    features=features,
                    action_taken=action_taken
                )
            except Exception:
                # Fallback explanation when SHAP/TensorFlow not available
                explanation = _generate_fallback_explanation(
                    decision_type, risk_score, action_taken, features
                )
            
            return jsonify(explanation)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def _generate_fallback_explanation(decision_type, risk_score, action_taken, features):
        """Generate explanation without SHAP"""
        # Calculate pseudo-contributions based on feature values
        contributions = {}
        feature_weights = {
            'incident_count': 0.25,
            'historical_risk': 0.20,
            'severity': 0.15,
            'hour': 0.10,
            'temperature': 0.08,
            'wind_speed': 0.07,
            'is_raining': 0.15
        }
        
        for name, weight in feature_weights.items():
            val = features.get(name, 0)
            if isinstance(val, bool):
                val = 1 if val else 0
            # Normalize and calculate contribution
            normalized = float(val) / 100 if name in ['incident_count', 'severity'] else float(val) / 24 if name == 'hour' else float(val)
            shap_val = normalized * weight * (1 if normalized > 0.5 else -0.5)
            contributions[name] = {
                'value': float(val) if not isinstance(features.get(name), bool) else int(val),
                'shap_value': round(shap_val, 4),
                'impact': 'positive' if shap_val > 0 else 'negative',
                'magnitude': round(abs(shap_val), 4)
            }
        
        # Sort by magnitude
        sorted_contributions = dict(sorted(
            contributions.items(),
            key=lambda x: x[1]['magnitude'],
            reverse=True
        ))
        
        risk_level = 'HIGH' if risk_score > 0.6 else 'MEDIUM' if risk_score > 0.3 else 'LOW'
        
        return {
            'decision_type': decision_type,
            'risk_score': risk_score,
            'action_taken': action_taken,
            'confidence': 0.85,
            'reasoning': [
                f'{risk_level} risk level ({risk_score*100:.0f}%) warranted preventive action',
                f'Primary risk driver: incident_count (contribution: {contributions.get("incident_count", {}).get("shap_value", 0):.3f})',
                'Signal timing adjustment recommended to reduce congestion'
            ],
            'human_summary': f'The system detected a {risk_level.lower()} risk level ({risk_score*100:.0f}%) primarily due to incident count and historical patterns. Action taken: {action_taken}. Confidence: 85%.',
            'prediction_explanation': {
                'feature_contributions': sorted_contributions,
                'prediction': risk_score
            }
        }
    
    # ==================== WEATHER ENDPOINTS ====================
    @app.route('/api/weather/current', methods=['GET'])
    def get_current_weather():
        """Get current weather data"""
        try:
            from data_acquisition.weather_api import OpenMeteoAPI
            
            api = OpenMeteoAPI()
            weather = api.fetch_current_weather()
            
            return jsonify(weather)
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    # ==================== DASHBOARD ENDPOINTS ====================
    @app.route('/api/dashboard/summary', methods=['GET'])
    def dashboard_summary():
        """Get dashboard summary data"""
        gold_data = app.config['DATA']['gold_data']
        alerts = app.config['DATA'].get('alerts', [])
        decisions = app.config['DATA'].get('decisions', [])
        
        if gold_data is None:
            gold_data = _generate_sample_data()
            app.config['DATA']['gold_data'] = gold_data
        
        risk_col = 'propagated_risk' if 'propagated_risk' in gold_data.columns else 'base_risk_score'
        
        # Handle NaN values for JSON serialization
        avg_risk = gold_data[risk_col].mean()
        avg_risk = 0.0 if pd.isna(avg_risk) else float(avg_risk)
        
        return jsonify({
            'total_zones': len(gold_data),
            'critical_zones': int(len(gold_data[gold_data[risk_col] > 0.8])),
            'high_risk_zones': int(len(gold_data[(gold_data[risk_col] > 0.6) & (gold_data[risk_col] <= 0.8)])),
            'medium_risk_zones': int(len(gold_data[(gold_data[risk_col] > 0.3) & (gold_data[risk_col] <= 0.6)])),
            'low_risk_zones': int(len(gold_data[gold_data[risk_col] <= 0.3])),
            'avg_risk': avg_risk,
            'total_incidents': int(gold_data['incident_count'].sum()) if 'incident_count' in gold_data.columns else 0,
            'active_alerts': len([a for a in alerts if not a.get('acknowledged', False)]),
            'pending_decisions': len(decisions),
            'last_update': app.config['DATA']['last_update'].isoformat() if app.config['DATA']['last_update'] else None
        })
    
    @app.route('/api/dashboard/trends', methods=['GET'])
    def dashboard_trends():
        """Get trend data for dashboard charts"""
        # Generate trend data based on actual risk scores
        gold_data = app.config['DATA']['gold_data']
        if gold_data is None:
            gold_data = _generate_sample_data()
            app.config['DATA']['gold_data'] = gold_data
        
        # Get base risk from data
        risk_col = 'propagated_risk' if 'propagated_risk' in gold_data.columns else 'base_risk_score'
        base_risk = float(gold_data[risk_col].mean()) if not gold_data.empty else 0.4
        base_risk = 0.4 if pd.isna(base_risk) else base_risk
        
        hours = 24
        now = datetime.now()
        np.random.seed(42)  # Consistent data
        
        trends = []
        for i in range(hours):
            time = now - timedelta(hours=hours-i-1)
            # Simulate variation around actual base risk
            variation = np.sin(i / 4) * 0.15 + np.random.uniform(-0.05, 0.05)
            risk = max(0, min(1, base_risk + variation))
            trends.append({
                'timestamp': time.isoformat(),
                'risk_score': float(round(risk, 3)),
                'incidents': int(np.random.randint(3, 15)),
                'alerts': int(np.random.randint(0, 5))
            })
        
        return jsonify({'trends': trends})
    
    return app


# ==================== HELPER FUNCTIONS ====================
def _load_gold_data():
    """Load real gold data from CSV files"""
    import glob
    import os
    
    gold_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'gold')
    files = glob.glob(os.path.join(gold_dir, 'gold_microzone_*.csv'))
    
    if files:
        latest_file = max(files, key=os.path.getctime)
        df = pd.read_csv(latest_file)
        print(f"Loaded {len(df)} zones from {latest_file}")
        
        # Add center coordinates if missing
        if 'center_lat' not in df.columns and 'h3_index' in df.columns:
            from data_engineering.h3_processor import H3Processor
            h3_proc = H3Processor()
            coords = df['h3_index'].apply(lambda h: h3_proc.h3_to_center(h) if pd.notna(h) else (None, None))
            df['center_lat'] = coords.apply(lambda x: x[0])
            df['center_lon'] = coords.apply(lambda x: x[1])
        
        return df
    return None
    
def _convert_to_serializable(obj):
    """Convert numpy/pandas types to JSON serializable types"""
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    else:
        return obj

def _generate_sample_data():
    """Generate sample data for demo purposes"""
    # First try to load real data
    real_data = _load_gold_data()
    if real_data is not None:
        return real_data
    
    # Fallback to generated sample data
    from data_engineering.h3_processor import H3Processor
    
    h3_proc = H3Processor()
    center_h3 = h3_proc.lat_lon_to_h3(40.7128, -74.0060)
    h3_indices = list(h3_proc.get_neighbors(center_h3, ring_size=3))[:50]
    
    np.random.seed(42)
    
    data = pd.DataFrame({
        'h3_index': h3_indices,
        'base_risk_score': np.random.uniform(0.1, 0.9, len(h3_indices)),
        'incident_count': np.random.randint(0, 50, len(h3_indices)),
        'total_severity': np.random.randint(0, 100, len(h3_indices)),
        'total_injured': np.random.randint(0, 20, len(h3_indices)),
        'center_lat': [h3_proc.h3_to_center(h)[0] for h in h3_indices],
        'center_lon': [h3_proc.h3_to_center(h)[1] for h in h3_indices]
    })
    
    data['risk_category'] = pd.cut(
        data['base_risk_score'],
        bins=[-np.inf, 0.3, 0.6, 0.8, np.inf],
        labels=['low', 'medium', 'high', 'critical']
    )
    
    return data


def run_server(host='0.0.0.0', port=5000, debug=True):
    """Run the Flask server"""
    app = create_app()
    print(f"\n{'='*60}")
    print("URBAN PULSE API SERVER")
    print(f"{'='*60}")
    print(f"Running on http://{host}:{port}")
    print(f"{'='*60}\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server()
