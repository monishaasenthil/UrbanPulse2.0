"""
NOVELTY 4: Human-in-the-Loop Explainability
SHAP-based explanations for every decision made by the system
"""
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import OUTPUTS_DIR

# SHAP import with error handling
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Using fallback explainability.")


class HumanInTheLoopExplainability:
    """
    NOVELTY 4: Human-in-the-Loop Explainability
    
    For every decision, the system generates:
    - SHAP explanations
    - Feature contributions
    - Why the action was taken
    - Human-readable summaries
    
    This enables transparency and trust in the AI system's decisions.
    """
    
    def __init__(self):
        self.explainer = None
        self.model = None
        self.feature_names = None
        self.explanation_history = []
        
    def initialize_explainer(self, model, X_background, feature_names=None):
        """
        Initialize SHAP explainer with a model
        
        Args:
            model: Trained model (must have predict method)
            X_background: Background data for SHAP (sample of training data)
            feature_names: List of feature names
            
        Returns:
            self
        """
        self.model = model
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_background.shape[1])]
        
        if SHAP_AVAILABLE:
            # Use appropriate explainer based on model type
            try:
                # Try TreeExplainer first (for tree-based models)
                self.explainer = shap.TreeExplainer(model)
            except:
                try:
                    # Fall back to KernelExplainer
                    X_sample = X_background[:100] if len(X_background) > 100 else X_background
                    self.explainer = shap.KernelExplainer(model.predict, X_sample)
                except:
                    print("Could not initialize SHAP explainer, using fallback")
                    self.explainer = None
        
        return self
    
    def explain_prediction(self, features, prediction=None, risk_score=None):
        """
        Explain a model prediction using SHAP
        
        Args:
            features: Feature dictionary or array
            prediction: Model prediction (optional)
            risk_score: Risk score (optional)
            
        Returns:
            Dictionary with explanation
        """
        if not self.model or not SHAP_AVAILABLE:
            return self._fallback_explanation(features, prediction, risk_score)
        
        try:
            # Convert features to array format
            if isinstance(features, dict):
                # Create array from feature dictionary
                X_array = np.zeros(len(self.feature_names))
                for i, name in enumerate(self.feature_names):
                    if name in features:
                        val = features[name]
                        # Handle non-numeric values
                        if isinstance(val, (int, float)):
                            X_array[i] = val
                        else:
                            X_array[i] = 0
                X_array = X_array.reshape(1, -1)
            else:
                X_array = np.array(features).reshape(1, -1)
            
            # Get prediction if not provided
            if prediction is None and self.model is not None:
                prediction = self.model.predict(X_array)[0]
            
            # Calculate SHAP values
            if self.explainer is not None:
                shap_values = self.explainer.shap_values(X_array)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                shap_values = shap_values.flatten()
            else:
                # Fallback: use feature values as proxy for importance
                shap_values = self._fallback_importance(X_array, features)
            
            # Create feature contributions
            contributions = {}
            for i, (name, shap_val) in enumerate(zip(self.feature_names, shap_values)):
                # Get feature value safely
                if isinstance(features, dict) and name in features:
                    val = features[name]
                elif i < len(X_array[0]):
                    val = X_array[0][i]
                else:
                    val = 0
                
                contributions[name] = {
                    'value': val,
                    'contribution': float(shap_val),
                    'magnitude': abs(float(shap_val)),
                    'impact': 'increases' if shap_val > 0 else 'decreases'
                }
            
            return {
                'prediction': float(prediction) if prediction is not None else None,
                'risk_score': float(risk_score) if risk_score is not None else None,
                'feature_contributions': contributions,
                'top_features': self._get_top_factors(contributions, 'positive', 5),
                'confidence': self._calculate_confidence(contributions)
            }
            
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return self._fallback_explanation(features, prediction, risk_score)
    
    def explain_decision(self, decision_type, risk_score, features, action_taken):
        """
        Generate human-readable explanation for a decision
        
        Args:
            decision_type: Type of decision (signal_change, routing, alert)
            risk_score: Risk score that triggered decision
            features: Feature values used
            action_taken: Description of action taken
            
        Returns:
            Dictionary with decision explanation
        """
        # Get prediction explanation
        pred_explanation = self.explain_prediction(features, risk_score)
        
        # Generate decision-specific explanation
        decision_explanation = {
            'decision_type': decision_type,
            'risk_score': risk_score,
            'action_taken': action_taken,
            'timestamp': datetime.now().isoformat(),
            'prediction_explanation': pred_explanation,
            'reasoning': self._generate_decision_reasoning(
                decision_type, risk_score, pred_explanation
            ),
            'confidence': self._calculate_confidence(pred_explanation),
            'human_summary': self._generate_human_summary(
                decision_type, risk_score, action_taken, pred_explanation
            )
        }
        
        return decision_explanation
    
    def _fallback_importance(self, X_array, feature_values):
        """Fallback importance calculation when SHAP is not available"""
        # Use normalized feature values as proxy
        if isinstance(X_array, np.ndarray) and X_array.size > 0:
            values = X_array[0]
        else:
            # Handle dict or empty array
            values = np.array(list(feature_values.values()) if isinstance(feature_values, dict) else [0])
        # Ensure numeric
        values = np.array([float(v) if isinstance(v, (int, float)) else 0 for v in values])
        # Normalize to [-1, 1] range
        max_abs = np.max(np.abs(values)) + 1e-6
        return values / max_abs * 0.5
    
    def _fallback_explanation(self, features, prediction=None, risk_score=None):
        """Fallback explanation when SHAP fails"""
        return {
            'prediction': float(prediction) if prediction is not None else 0.5,
            'risk_score': float(risk_score) if risk_score is not None else 0.5,
            'feature_contributions': {},
            'top_features': [],
            'confidence': 0.5,
            'fallback': True
        }
    
    def _get_top_factors(self, contributions, impact_type, n=5):
        """Get top N factors of specified impact type"""
        filtered = [
            (name, data) for name, data in contributions.items()
            if data['impact'] == impact_type
        ]
        sorted_factors = sorted(filtered, key=lambda x: x[1]['magnitude'], reverse=True)
        return [
            {'feature': name, 'value': data['value'], 'contribution': data['shap_value']}
            for name, data in sorted_factors[:n]
        ]
    
    def _generate_summary(self, prediction, contributions):
        """Generate text summary of prediction"""
        top_features = list(contributions.items())[:3]
        
        summary_parts = []
        
        if prediction is not None:
            risk_level = 'high' if prediction > 0.7 else 'medium' if prediction > 0.4 else 'low'
            summary_parts.append(f"Risk Level: {risk_level.upper()} ({prediction:.2f})")
        
        summary_parts.append("\nKey Contributing Factors:")
        for name, data in top_features:
            direction = "increases" if data['shap_value'] > 0 else "decreases"
            summary_parts.append(
                f"  • {name} = {data['value']:.2f} → {direction} risk by {abs(data['shap_value']):.3f}"
            )
        
        return "\n".join(summary_parts)
    
    def _generate_decision_reasoning(self, decision_type, risk_score, explanation):
        """Generate reasoning for why decision was made"""
        reasoning = []
        
        # Risk threshold reasoning
        if risk_score > 0.8:
            reasoning.append(f"CRITICAL risk level ({risk_score:.2f}) triggered immediate action")
        elif risk_score > 0.6:
            reasoning.append(f"HIGH risk level ({risk_score:.2f}) warranted preventive action")
        elif risk_score > 0.4:
            reasoning.append(f"MODERATE risk level ({risk_score:.2f}) suggested precautionary measures")
        
        # Top factor reasoning
        top_positive = explanation.get('top_positive_factors', [])
        if top_positive:
            top = top_positive[0]
            reasoning.append(
                f"Primary risk driver: {top['feature']} (contribution: {top['contribution']:.3f})"
            )
        
        # Decision type specific reasoning
        if decision_type == 'signal_change':
            reasoning.append("Signal timing adjustment recommended to reduce congestion")
        elif decision_type == 'emergency_routing':
            reasoning.append("Alternative routing suggested to avoid high-risk zone")
        elif decision_type == 'alert':
            reasoning.append("Alert issued to notify relevant authorities")
        
        return reasoning
    
    def _calculate_confidence(self, explanation):
        """Calculate confidence in the explanation"""
        contributions = explanation.get('feature_contributions', {})
        
        if not contributions:
            return 0.5
        
        # Higher confidence if top features have high magnitude
        magnitudes = [data['magnitude'] for data in contributions.values()]
        top_magnitude = max(magnitudes) if magnitudes else 0
        
        # Confidence based on how much top features explain
        total_magnitude = sum(magnitudes)
        top_3_magnitude = sum(sorted(magnitudes, reverse=True)[:3])
        
        concentration = top_3_magnitude / (total_magnitude + 1e-6)
        
        return min(0.5 + concentration * 0.5, 1.0)
    
    def _generate_human_summary(self, decision_type, risk_score, action_taken, explanation):
        """Generate human-readable summary"""
        risk_level = 'critical' if risk_score > 0.8 else 'high' if risk_score > 0.6 else 'moderate' if risk_score > 0.4 else 'low'
        
        top_factors = explanation.get('top_positive_factors', [])[:2]
        factor_text = ""
        if top_factors:
            factors = [f['feature'].replace('_', ' ') for f in top_factors]
            factor_text = f" primarily due to {' and '.join(factors)}"
        
        summary = (
            f"The system detected a {risk_level} risk level ({risk_score:.0%}){factor_text}. "
            f"In response, the following action was taken: {action_taken}. "
            f"This decision was made with {self._calculate_confidence(explanation):.0%} confidence."
        )
        
        return summary
    
    def get_feature_importance_summary(self):
        """
        Get aggregated feature importance across all explanations
        
        Returns:
            DataFrame with feature importance statistics
        """
        if not self.explanation_history:
            return pd.DataFrame()
        
        importance_data = {}
        
        for explanation in self.explanation_history:
            for feature, data in explanation.get('feature_contributions', {}).items():
                if feature not in importance_data:
                    importance_data[feature] = []
                importance_data[feature].append(abs(data['shap_value']))
        
        summary = []
        for feature, values in importance_data.items():
            summary.append({
                'feature': feature,
                'mean_importance': np.mean(values),
                'std_importance': np.std(values),
                'max_importance': max(values),
                'count': len(values)
            })
        
        df = pd.DataFrame(summary)
        return df.sort_values('mean_importance', ascending=False)
    
    def export_explanations(self, filepath=None):
        """
        Export all explanations to file
        
        Args:
            filepath: Path to save (default: outputs/explanations.json)
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            filepath = os.path.join(OUTPUTS_DIR, 'explanations.json')
        
        with open(filepath, 'w') as f:
            json.dump(self.explanation_history, f, indent=2, default=str)
        
        print(f"Explanations exported to {filepath}")
        return filepath
    
    def generate_report(self, explanation):
        """
        Generate a formatted report for an explanation
        
        Args:
            explanation: Explanation dictionary
            
        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 60)
        report.append("URBAN PULSE - DECISION EXPLANATION REPORT")
        report.append("=" * 60)
        
        if 'decision_type' in explanation:
            report.append(f"\nDecision Type: {explanation['decision_type'].upper()}")
            report.append(f"Action Taken: {explanation.get('action_taken', 'N/A')}")
        
        report.append(f"Timestamp: {explanation.get('timestamp', 'N/A')}")
        
        pred_exp = explanation.get('prediction_explanation', explanation)
        
        if pred_exp.get('prediction') is not None:
            report.append(f"\nRisk Score: {pred_exp['prediction']:.3f}")
        
        report.append("\n" + "-" * 60)
        report.append("FEATURE CONTRIBUTIONS")
        report.append("-" * 60)
        
        contributions = pred_exp.get('feature_contributions', {})
        for i, (feature, data) in enumerate(list(contributions.items())[:10]):
            bar_length = int(abs(data['shap_value']) * 20)
            bar = "+" * bar_length if data['shap_value'] > 0 else "-" * bar_length
            report.append(
                f"{feature:30s} | {data['value']:8.2f} | {bar:20s} | {data['shap_value']:+.3f}"
            )
        
        if 'reasoning' in explanation:
            report.append("\n" + "-" * 60)
            report.append("REASONING")
            report.append("-" * 60)
            for reason in explanation['reasoning']:
                report.append(f"  • {reason}")
        
        if 'human_summary' in explanation:
            report.append("\n" + "-" * 60)
            report.append("SUMMARY")
            report.append("-" * 60)
            report.append(explanation['human_summary'])
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def test_explainability():
    """Test explainability functionality"""
    print("Testing Human-in-the-Loop Explainability...")
    print("=" * 70)
    
    # Create sample model and data
    from sklearn.ensemble import GradientBoostingRegressor
    
    np.random.seed(42)
    n_samples = 200
    
    feature_names = ['hour', 'is_raining', 'incident_count', 'severity', 
                     'temperature', 'wind_speed', 'historical_risk']
    
    X = pd.DataFrame({
        'hour': np.random.randint(0, 24, n_samples),
        'is_raining': np.random.choice([0, 1], n_samples),
        'incident_count': np.random.randint(0, 50, n_samples),
        'severity': np.random.randint(0, 20, n_samples),
        'temperature': np.random.uniform(0, 35, n_samples),
        'wind_speed': np.random.uniform(0, 50, n_samples),
        'historical_risk': np.random.uniform(0, 1, n_samples)
    })
    
    y = (0.3 * X['incident_count'] / 50 + 
         0.2 * X['severity'] / 20 + 
         0.2 * X['is_raining'] +
         0.15 * X['historical_risk'] +
         0.15 * (X['hour'].isin([7,8,9,16,17,18])).astype(int) +
         np.random.uniform(0, 0.1, n_samples))
    
    # Train model
    model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Initialize explainer
    explainer = HumanInTheLoopExplainability()
    explainer.initialize_explainer(model, X, feature_names)
    
    # Test 1: Explain single prediction
    print("\n1. Single Prediction Explanation:")
    sample = X.iloc[0]
    explanation = explainer.explain_prediction(sample)
    print(explanation['summary'])
    
    # Test 2: Explain decision
    print("\n2. Decision Explanation:")
    decision_exp = explainer.explain_decision(
        decision_type='signal_change',
        risk_score=0.75,
        features=sample,
        action_taken='Extend green light duration by 15 seconds on Main St'
    )
    print(f"   Human Summary: {decision_exp['human_summary']}")
    
    # Test 3: Generate report
    print("\n3. Full Report:")
    report = explainer.generate_report(decision_exp)
    print(report)
    
    # Test 4: Feature importance summary
    print("\n4. Generating multiple explanations...")
    for i in range(10):
        explainer.explain_prediction(X.iloc[i])
    
    importance_df = explainer.get_feature_importance_summary()
    print("\nFeature Importance Summary:")
    print(importance_df.head())
    
    print("\n" + "=" * 70)
    print("Explainability Test Complete!")
    
    return explainer


if __name__ == "__main__":
    test_explainability()
