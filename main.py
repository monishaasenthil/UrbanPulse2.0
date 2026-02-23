"""
Urban Pulse 2.0 - Main Orchestrator
Complete end-to-end intelligent traffic risk management system
"""
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_acquisition.data_fetcher import DataFetcher
from data_engineering.pipeline import DataPipeline
from features.feature_engineer import FeatureEngineer
from models.model_trainer import ModelTrainer
from novelties.cars import ContextAdaptiveRiskScoring
from novelties.risk_propagation import PriorityAwareRiskPropagation
from novelties.feedback_loop import ActionImpactFeedbackLoop
from novelties.explainability import HumanInTheLoopExplainability
from decision_engine.decision_engine import DecisionEngine


class UrbanPulse:
    """
    Urban Pulse 2.0 - Main System Orchestrator
    
    Integrates all components:
    - Data Acquisition (NYC Collisions + Open-Meteo APIs)
    - Data Engineering (Bronze/Silver/Gold layers)
    - Feature Engineering
    - ML Models (with fine-tuning)
    - 4 Novel Contributions:
        1. CARS (Context Aware Adaptive Risk Scoring)
        2. Risk Propagation Network
        3. Action Impact Feedback Loop
        4. Human-in-the-Loop Explainability
    - Decision Engine
    """
    
    def __init__(self):
        print("\n" + "=" * 70)
        print("URBAN PULSE 2.0 - INTELLIGENT TRAFFIC RISK MANAGEMENT")
        print("=" * 70)
        print("\nInitializing system components...")
        
        # Core components
        self.data_fetcher = DataFetcher()
        self.pipeline = DataPipeline()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        
        # Novelty components
        self.cars = ContextAdaptiveRiskScoring()
        self.risk_propagator = PriorityAwareRiskPropagation()
        self.feedback_loop = ActionImpactFeedbackLoop()
        self.explainer = HumanInTheLoopExplainability()
        
        # Decision engine
        self.decision_engine = DecisionEngine()
        
        # State
        self.gold_data = None
        self.model_trained = False
        self.last_run = None
        
        print("✓ All components initialized")
    
    def run_full_pipeline(self, days_back=30, collision_limit=5000):
        """
        Run the complete Urban Pulse pipeline
        
        Args:
            days_back: Days of historical data to fetch
            collision_limit: Maximum collision records
            
        Returns:
            Dictionary with all results
        """
        print("\n" + "=" * 70)
        print("RUNNING FULL URBAN PULSE PIPELINE")
        print("=" * 70)
        
        start_time = datetime.now()
        results = {}
        
        # ============================================================
        # PHASE 1: DATA ACQUISITION & ENGINEERING
        # ============================================================
        print("\n" + "-" * 70)
        print("PHASE 1: DATA ACQUISITION & ENGINEERING")
        print("-" * 70)
        
        pipeline_results = self.pipeline.run_full_pipeline(
            days_back=days_back,
            collision_limit=collision_limit
        )
        
        self.gold_data = pipeline_results.get('gold_microzone')
        results['data_pipeline'] = {
            'records': len(self.gold_data) if self.gold_data is not None else 0,
            'status': 'success' if self.gold_data is not None else 'failed'
        }
        
        if self.gold_data is None or self.gold_data.empty:
            print("WARNING: No data available. Using sample data for demo.")
            self.gold_data = self._generate_sample_data()
        
        # ============================================================
        # PHASE 2: FEATURE ENGINEERING
        # ============================================================
        print("\n" + "-" * 70)
        print("PHASE 2: FEATURE ENGINEERING")
        print("-" * 70)
        
        merged_data = pipeline_results.get('merged')
        if merged_data is not None and not merged_data.empty:
            featured_data = self.feature_engineer.engineer_features(merged_data, self.gold_data)
            results['features'] = {
                'records': len(featured_data),
                'feature_count': len(featured_data.columns)
            }
        else:
            featured_data = None
            results['features'] = {'status': 'skipped - no merged data'}
        
        # ============================================================
        # PHASE 3: MODEL TRAINING
        # ============================================================
        print("\n" + "-" * 70)
        print("PHASE 3: MODEL TRAINING")
        print("-" * 70)
        
        if featured_data is not None and not featured_data.empty:
            X, y, feature_names = self.feature_engineer.prepare_ml_dataset(featured_data)
            
            if len(X) > 50:
                training_results = self.model_trainer.train_all_models(
                    X, y, tune_hyperparameters=False
                )
                self.model_trained = True
                results['models'] = {
                    'best_model': self.model_trainer.best_model,
                    'metrics': self.model_trainer.best_metrics
                }
            else:
                print("Insufficient data for model training")
                results['models'] = {'status': 'skipped - insufficient data'}
        else:
            results['models'] = {'status': 'skipped - no featured data'}
        
        # ============================================================
        # PHASE 4: NOVELTY 1 - CARS (Context Adaptive Risk Scoring)
        # ============================================================
        print("\n" + "-" * 70)
        print("PHASE 4: NOVELTY 1 - CONTEXT ADAPTIVE RISK SCORING (CARS)")
        print("-" * 70)
        
        self.gold_data = self.cars.calculate_batch_scores(self.gold_data)
        
        cars_summary = {
            'zones_scored': len(self.gold_data),
            'avg_cars_score': self.gold_data['cars_score'].mean(),
            'contexts': self.gold_data['cars_context'].value_counts().to_dict()
        }
        results['novelty_1_cars'] = cars_summary
        print(f"✓ CARS scores calculated for {cars_summary['zones_scored']} zones")
        print(f"  Average CARS score: {cars_summary['avg_cars_score']:.3f}")
        
        # ============================================================
        # PHASE 5: NOVELTY 2 - Risk Propagation
        # ============================================================
        print("\n" + "-" * 70)
        print("PHASE 5: NOVELTY 2 - PRIORITY AWARE RISK PROPAGATION")
        print("-" * 70)
        
        self.gold_data = self.risk_propagator.calculate_propagated_scores(
            self.gold_data, risk_col='cars_score'
        )
        
        propagation_summary = self.risk_propagator.get_propagation_summary()
        results['novelty_2_propagation'] = propagation_summary
        print(f"✓ Risk propagated across {propagation_summary.get('total_zones', 0)} zones")
        print(f"  Zones amplified: {propagation_summary.get('zones_amplified', 0)}")
        
        # ============================================================
        # PHASE 6: DECISION ENGINE
        # ============================================================
        print("\n" + "-" * 70)
        print("PHASE 6: DECISION ENGINE")
        print("-" * 70)
        
        # Get current weather for decisions
        weather_data = None
        try:
            from data_acquisition.weather_api import OpenMeteoAPI
            weather_api = OpenMeteoAPI()
            weather_data = weather_api.fetch_current_weather()
        except:
            weather_data = {'precipitation': 0, 'wind_speed': 10, 'visibility': 10000}
        
        decision_results = self.decision_engine.process_risk_data(self.gold_data, weather_data)
        results['decisions'] = {
            'signal_plans': len(decision_results.get('signal_plan', [])),
            'alerts': len(decision_results.get('alerts', [])),
            'priority_decisions': len(decision_results.get('priority_decisions', []))
        }
        
        # ============================================================
        # PHASE 7: NOVELTY 3 - Feedback Loop (Simulation)
        # ============================================================
        print("\n" + "-" * 70)
        print("PHASE 7: NOVELTY 3 - ACTION IMPACT FEEDBACK LOOP")
        print("-" * 70)
        
        feedback_results = self.feedback_loop.simulate_feedback_cycle(n_actions=20)
        results['novelty_3_feedback'] = feedback_results['summary']
        print(f"✓ Feedback loop simulated with {feedback_results['actions_simulated']} actions")
        print(f"  Success rate: {feedback_results['summary']['success_rate']:.1%}")
        
        # ============================================================
        # PHASE 8: NOVELTY 4 - Explainability
        # ============================================================
        print("\n" + "-" * 70)
        print("PHASE 8: NOVELTY 4 - HUMAN-IN-THE-LOOP EXPLAINABILITY")
        print("-" * 70)
        
        # Generate sample explanation
        if self.model_trained and featured_data is not None:
            self.explainer.initialize_explainer(
                self.model_trainer.get_best_model().model,
                X[:100],
                feature_names
            )
        
        sample_explanation = self.explainer.explain_decision(
            decision_type='signal_change',
            risk_score=0.75,
            features={'incident_count': 25, 'severity': 15, 'hour': 14},
            action_taken='Extend green light by 15 seconds'
        )
        
        results['novelty_4_explainability'] = {
            'confidence': sample_explanation.get('confidence', 0.85),
            'reasoning_count': len(sample_explanation.get('reasoning', []))
        }
        print("✓ Explainability engine ready")
        print(f"  Sample explanation confidence: {sample_explanation.get('confidence', 0.85):.1%}")
        
        # ============================================================
        # EXPORT OUTPUTS
        # ============================================================
        print("\n" + "-" * 70)
        print("EXPORTING OUTPUTS")
        print("-" * 70)
        
        output_paths = self.decision_engine.export_all_outputs()
        results['outputs'] = output_paths
        
        # ============================================================
        # SUMMARY
        # ============================================================
        elapsed = datetime.now() - start_time
        self.last_run = datetime.now()
        
        print("\n" + "=" * 70)
        print("URBAN PULSE PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nTotal execution time: {elapsed}")
        print(f"\nSummary:")
        print(f"  • Zones processed: {len(self.gold_data)}")
        print(f"  • Critical zones: {len(self.gold_data[self.gold_data['propagated_risk'] > 0.8])}")
        print(f"  • Decisions generated: {results['decisions']['priority_decisions']}")
        print(f"  • Alerts created: {results['decisions']['alerts']}")
        
        return results
    
    def _generate_sample_data(self):
        """Generate sample data for demo purposes"""
        import pandas as pd
        import numpy as np
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
            'hour': np.random.randint(0, 24, len(h3_indices)),
            'is_raining': np.random.choice([0, 1], len(h3_indices), p=[0.8, 0.2]),
            'precipitation': np.random.exponential(1, len(h3_indices)),
            'wind_speed': np.random.uniform(0, 40, len(h3_indices)),
            'visibility': np.random.uniform(1000, 10000, len(h3_indices)),
            'center_lat': [h3_proc.h3_to_center(h)[0] for h in h3_indices],
            'center_lon': [h3_proc.h3_to_center(h)[1] for h in h3_indices]
        })
        
        data['risk_category'] = pd.cut(
            data['base_risk_score'],
            bins=[-np.inf, 0.3, 0.6, 0.8, np.inf],
            labels=['low', 'medium', 'high', 'critical']
        )
        
        return data
    
    def get_status(self):
        """Get current system status"""
        return {
            'initialized': True,
            'data_loaded': self.gold_data is not None,
            'zones': len(self.gold_data) if self.gold_data is not None else 0,
            'model_trained': self.model_trained,
            'last_run': self.last_run.isoformat() if self.last_run else None
        }


def main():
    """Main entry point"""
    urban_pulse = UrbanPulse()
    
    # Run full pipeline
    results = urban_pulse.run_full_pipeline(days_back=7, collision_limit=2000)
    
    # Print final status
    print("\n" + "=" * 70)
    print("SYSTEM STATUS")
    print("=" * 70)
    status = urban_pulse.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    return urban_pulse, results


if __name__ == "__main__":
    main()
