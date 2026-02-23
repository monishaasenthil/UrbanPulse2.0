"""
NOVELTY 3: Action Impact Feedback Loop
Closed-loop learning system that measures action effectiveness and updates models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import FEEDBACK_LEARNING_RATE, FEEDBACK_WINDOW_HOURS, OUTPUTS_DIR


class ActionImpactFeedbackLoop:
    """
    NOVELTY 3: Action Impact Feedback Loop
    
    Closed-loop learning system:
    Predict → Act → Measure → Learn
    
    After every action:
    - Check congestion reduction
    - Compute effectiveness score
    - Update model weights
    - Improve future predictions
    
    This creates a continuously improving system that learns
    from the outcomes of its own decisions.
    """
    
    def __init__(self, learning_rate=FEEDBACK_LEARNING_RATE, 
                 feedback_window_hours=FEEDBACK_WINDOW_HOURS):
        """
        Initialize feedback loop
        
        Args:
            learning_rate: Rate at which to update weights
            feedback_window_hours: Hours to wait before measuring impact
        """
        self.learning_rate = learning_rate
        self.feedback_window_hours = feedback_window_hours
        
        # Action history
        self.actions = []
        self.action_outcomes = []
        
        # Effectiveness tracking
        self.effectiveness_history = defaultdict(list)
        
        # Weight adjustments
        self.weight_adjustments = {}
        
        # Metrics
        self.metrics = {
            'total_actions': 0,
            'successful_actions': 0,
            'average_effectiveness': 0,
            'learning_iterations': 0
        }
        
    def record_action(self, action_id, action_type, zone_id, 
                      predicted_risk, action_details, timestamp=None):
        """
        Record an action taken by the system
        
        Args:
            action_id: Unique identifier for the action
            action_type: Type of action (signal_change, routing, alert)
            zone_id: H3 zone where action was taken
            predicted_risk: Risk score that triggered the action
            action_details: Dictionary with action specifics
            timestamp: When action was taken (default: now)
            
        Returns:
            Action record
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        action = {
            'action_id': action_id,
            'action_type': action_type,
            'zone_id': zone_id,
            'predicted_risk': predicted_risk,
            'action_details': action_details,
            'timestamp': timestamp,
            'status': 'pending_evaluation',
            'outcome': None
        }
        
        self.actions.append(action)
        self.metrics['total_actions'] += 1
        
        return action
    
    def record_outcome(self, action_id, observed_incidents, observed_severity,
                       congestion_before, congestion_after, response_time=None):
        """
        Record the outcome after an action
        
        Args:
            action_id: ID of the action to evaluate
            observed_incidents: Number of incidents after action
            observed_severity: Total severity after action
            congestion_before: Congestion level before action
            congestion_after: Congestion level after action
            response_time: Emergency response time (if applicable)
            
        Returns:
            Outcome record with effectiveness score
        """
        # Find the action
        action = None
        for a in self.actions:
            if a['action_id'] == action_id:
                action = a
                break
        
        if action is None:
            raise ValueError(f"Action {action_id} not found")
        
        # Calculate effectiveness metrics
        congestion_reduction = (congestion_before - congestion_after) / max(congestion_before, 1)
        
        # Effectiveness score (0-1)
        effectiveness = self._calculate_effectiveness(
            action['action_type'],
            action['predicted_risk'],
            observed_incidents,
            observed_severity,
            congestion_reduction,
            response_time
        )
        
        outcome = {
            'action_id': action_id,
            'observed_incidents': observed_incidents,
            'observed_severity': observed_severity,
            'congestion_before': congestion_before,
            'congestion_after': congestion_after,
            'congestion_reduction': congestion_reduction,
            'response_time': response_time,
            'effectiveness': effectiveness,
            'evaluation_timestamp': datetime.now()
        }
        
        # Update action status
        action['status'] = 'evaluated'
        action['outcome'] = outcome
        
        self.action_outcomes.append(outcome)
        
        # Track effectiveness by action type
        self.effectiveness_history[action['action_type']].append(effectiveness)
        
        # Update metrics
        if effectiveness > 0.5:
            self.metrics['successful_actions'] += 1
        
        self._update_average_effectiveness()
        
        return outcome
    
    def _calculate_effectiveness(self, action_type, predicted_risk, 
                                  observed_incidents, observed_severity,
                                  congestion_reduction, response_time):
        """
        Calculate effectiveness score for an action
        
        Args:
            Various metrics about the action and its outcome
            
        Returns:
            Effectiveness score (0-1)
        """
        effectiveness = 0.5  # Start neutral
        
        # Congestion reduction component (40% weight)
        if congestion_reduction > 0.2:
            effectiveness += 0.2
        elif congestion_reduction > 0.1:
            effectiveness += 0.1
        elif congestion_reduction < -0.1:
            effectiveness -= 0.1
        
        # Incident reduction component (30% weight)
        # Compare to expected based on predicted risk
        expected_incidents = predicted_risk * 10  # Rough estimate
        if observed_incidents < expected_incidents * 0.7:
            effectiveness += 0.15
        elif observed_incidents < expected_incidents:
            effectiveness += 0.05
        elif observed_incidents > expected_incidents * 1.3:
            effectiveness -= 0.1
        
        # Severity component (20% weight)
        expected_severity = predicted_risk * 20
        if observed_severity < expected_severity * 0.7:
            effectiveness += 0.1
        elif observed_severity > expected_severity * 1.3:
            effectiveness -= 0.1
        
        # Response time component for emergency actions (10% weight)
        if response_time is not None and action_type == 'emergency_routing':
            if response_time < 5:  # Under 5 minutes
                effectiveness += 0.1
            elif response_time < 10:
                effectiveness += 0.05
            elif response_time > 15:
                effectiveness -= 0.05
        
        return np.clip(effectiveness, 0, 1)
    
    def learn_from_feedback(self, min_samples=10):
        """
        Update model weights based on accumulated feedback
        
        Args:
            min_samples: Minimum samples needed before learning
            
        Returns:
            Dictionary with learning results
        """
        if len(self.action_outcomes) < min_samples:
            return {'status': 'insufficient_data', 'samples': len(self.action_outcomes)}
        
        learning_results = {}
        
        # Analyze effectiveness by action type
        for action_type, effectiveness_list in self.effectiveness_history.items():
            if len(effectiveness_list) < 3:
                continue
            
            avg_effectiveness = np.mean(effectiveness_list)
            recent_effectiveness = np.mean(effectiveness_list[-10:])
            
            # Calculate weight adjustment
            if recent_effectiveness > 0.6:
                # Action type is working well, increase confidence
                adjustment = self.learning_rate * (recent_effectiveness - 0.5)
            elif recent_effectiveness < 0.4:
                # Action type not working, decrease confidence
                adjustment = -self.learning_rate * (0.5 - recent_effectiveness)
            else:
                adjustment = 0
            
            self.weight_adjustments[action_type] = {
                'adjustment': adjustment,
                'avg_effectiveness': avg_effectiveness,
                'recent_effectiveness': recent_effectiveness,
                'sample_count': len(effectiveness_list)
            }
            
            learning_results[action_type] = self.weight_adjustments[action_type]
        
        self.metrics['learning_iterations'] += 1
        
        return {
            'status': 'learned',
            'iterations': self.metrics['learning_iterations'],
            'adjustments': learning_results
        }
    
    def get_action_recommendation_weight(self, action_type):
        """
        Get weight adjustment for an action type based on feedback
        
        Args:
            action_type: Type of action
            
        Returns:
            Weight multiplier (1.0 = no change)
        """
        if action_type not in self.weight_adjustments:
            return 1.0
        
        adjustment = self.weight_adjustments[action_type].get('adjustment', 0)
        return 1.0 + adjustment
    
    def should_take_action(self, action_type, risk_score, threshold=0.6):
        """
        Determine if an action should be taken based on feedback learning
        
        Args:
            action_type: Proposed action type
            risk_score: Current risk score
            threshold: Base threshold for action
            
        Returns:
            Boolean and adjusted threshold
        """
        weight = self.get_action_recommendation_weight(action_type)
        adjusted_threshold = threshold / weight  # Lower threshold if action is effective
        
        return risk_score >= adjusted_threshold, adjusted_threshold
    
    def _update_average_effectiveness(self):
        """Update running average effectiveness"""
        if self.action_outcomes:
            self.metrics['average_effectiveness'] = np.mean(
                [o['effectiveness'] for o in self.action_outcomes]
            )
    
    def get_feedback_summary(self):
        """
        Get summary of feedback loop performance
        
        Returns:
            Dictionary with performance metrics
        """
        summary = {
            'total_actions': self.metrics['total_actions'],
            'evaluated_actions': len(self.action_outcomes),
            'successful_actions': self.metrics['successful_actions'],
            'success_rate': (self.metrics['successful_actions'] / 
                           max(len(self.action_outcomes), 1)),
            'average_effectiveness': self.metrics['average_effectiveness'],
            'learning_iterations': self.metrics['learning_iterations'],
            'action_type_performance': {}
        }
        
        for action_type, effectiveness_list in self.effectiveness_history.items():
            if effectiveness_list:
                summary['action_type_performance'][action_type] = {
                    'count': len(effectiveness_list),
                    'avg_effectiveness': np.mean(effectiveness_list),
                    'trend': 'improving' if len(effectiveness_list) > 5 and 
                            np.mean(effectiveness_list[-5:]) > np.mean(effectiveness_list[:5])
                            else 'stable'
                }
        
        return summary
    
    def get_improvement_recommendations(self):
        """
        Generate recommendations for system improvement
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        for action_type, data in self.weight_adjustments.items():
            if data['recent_effectiveness'] < 0.4:
                recommendations.append({
                    'action_type': action_type,
                    'issue': 'low_effectiveness',
                    'recommendation': f"Consider revising {action_type} strategy. "
                                     f"Recent effectiveness: {data['recent_effectiveness']:.2f}",
                    'priority': 'high'
                })
            elif data['recent_effectiveness'] > 0.7:
                recommendations.append({
                    'action_type': action_type,
                    'issue': 'high_effectiveness',
                    'recommendation': f"{action_type} is performing well. "
                                     f"Consider expanding its use.",
                    'priority': 'low'
                })
        
        return recommendations
    
    def export_feedback_data(self, filepath=None):
        """
        Export feedback data for analysis
        
        Args:
            filepath: Path to save data (default: outputs/feedback_data.json)
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            filepath = os.path.join(OUTPUTS_DIR, 'feedback_data.json')
        
        data = {
            'actions': [
                {**a, 'timestamp': str(a['timestamp'])} 
                for a in self.actions
            ],
            'outcomes': [
                {**o, 'evaluation_timestamp': str(o['evaluation_timestamp'])}
                for o in self.action_outcomes
            ],
            'metrics': self.metrics,
            'weight_adjustments': self.weight_adjustments,
            'summary': self.get_feedback_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Feedback data exported to {filepath}")
        return filepath
    
    def simulate_feedback_cycle(self, n_actions=20):
        """
        Simulate a feedback cycle for testing
        
        Args:
            n_actions: Number of actions to simulate
            
        Returns:
            Simulation results
        """
        np.random.seed(42)
        
        action_types = ['signal_change', 'emergency_routing', 'alert']
        
        for i in range(n_actions):
            # Simulate action
            action_type = np.random.choice(action_types)
            zone_id = f"zone_{i % 10}"
            predicted_risk = np.random.uniform(0.5, 1.0)
            
            action = self.record_action(
                action_id=f"action_{i}",
                action_type=action_type,
                zone_id=zone_id,
                predicted_risk=predicted_risk,
                action_details={'simulated': True}
            )
            
            # Simulate outcome (with some correlation to action type effectiveness)
            base_effectiveness = {
                'signal_change': 0.6,
                'emergency_routing': 0.7,
                'alert': 0.5
            }
            
            congestion_before = np.random.uniform(50, 100)
            reduction_factor = base_effectiveness[action_type] + np.random.uniform(-0.2, 0.2)
            congestion_after = congestion_before * (1 - reduction_factor * 0.3)
            
            self.record_outcome(
                action_id=f"action_{i}",
                observed_incidents=int(predicted_risk * 5 * np.random.uniform(0.5, 1.5)),
                observed_severity=int(predicted_risk * 10 * np.random.uniform(0.5, 1.5)),
                congestion_before=congestion_before,
                congestion_after=congestion_after,
                response_time=np.random.uniform(3, 15) if action_type == 'emergency_routing' else None
            )
        
        # Learn from feedback
        learning_results = self.learn_from_feedback(min_samples=5)
        
        return {
            'actions_simulated': n_actions,
            'learning_results': learning_results,
            'summary': self.get_feedback_summary()
        }


def test_feedback_loop():
    """Test feedback loop functionality"""
    print("Testing Action Impact Feedback Loop...")
    print("=" * 70)
    
    feedback = ActionImpactFeedbackLoop(learning_rate=0.1)
    
    # Run simulation
    print("\n1. Running feedback simulation...")
    results = feedback.simulate_feedback_cycle(n_actions=30)
    
    print(f"   Actions simulated: {results['actions_simulated']}")
    print(f"   Learning status: {results['learning_results']['status']}")
    
    # Summary
    print("\n2. Feedback Summary:")
    summary = results['summary']
    print(f"   Total actions: {summary['total_actions']}")
    print(f"   Success rate: {summary['success_rate']:.2%}")
    print(f"   Average effectiveness: {summary['average_effectiveness']:.3f}")
    
    # Action type performance
    print("\n3. Performance by Action Type:")
    for action_type, perf in summary['action_type_performance'].items():
        print(f"   {action_type}:")
        print(f"      Count: {perf['count']}")
        print(f"      Avg effectiveness: {perf['avg_effectiveness']:.3f}")
        print(f"      Trend: {perf['trend']}")
    
    # Weight adjustments
    print("\n4. Weight Adjustments:")
    for action_type, adj in feedback.weight_adjustments.items():
        weight = feedback.get_action_recommendation_weight(action_type)
        print(f"   {action_type}: weight multiplier = {weight:.3f}")
    
    # Recommendations
    print("\n5. Improvement Recommendations:")
    recommendations = feedback.get_improvement_recommendations()
    for rec in recommendations:
        print(f"   [{rec['priority'].upper()}] {rec['recommendation']}")
    
    print("\n" + "=" * 70)
    print("Feedback Loop Test Complete!")
    
    return feedback


if __name__ == "__main__":
    test_feedback_loop()
