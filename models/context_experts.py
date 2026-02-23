"""
Context Expert Ensemble
Trains separate models for different contexts (morning peak, evening peak, night, rainy)
Novel ML Enhancement: Ensemble of Context Experts
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODELS_DIR, CONTEXT_WINDOWS
from models.base_models import GradientBoostingModel


class ContextExpertEnsemble:
    """
    Ensemble of Context-Aware Expert Models
    
    Instead of ONE global model, trains separate models for:
    - Morning peak hours
    - Evening peak hours
    - Night time
    - Rainy conditions
    - Normal conditions (default)
    
    This is a NOVEL ML ENHANCEMENT that improves prediction accuracy
    by specializing models for different traffic contexts.
    """
    
    def __init__(self):
        self.experts = {}
        self.context_windows = CONTEXT_WINDOWS
        self.is_fitted = False
        self.default_expert = None
        
        # Define contexts
        self.contexts = ['morning_peak', 'evening_peak', 'night', 'rainy', 'normal']
        
    def _get_context(self, row):
        """
        Determine context for a data point
        
        Args:
            row: DataFrame row with hour and weather info
            
        Returns:
            Context string
        """
        hour = row.get('hour', 12)
        is_raining = row.get('is_raining', False) or row.get('precipitation', 0) > 0
        
        # Rainy takes priority
        if is_raining:
            return 'rainy'
        
        # Check time-based contexts
        morning_start, morning_end = self.context_windows['morning_peak']
        evening_start, evening_end = self.context_windows['evening_peak']
        night_start, night_end = self.context_windows['night']
        
        if morning_start <= hour < morning_end:
            return 'morning_peak'
        elif evening_start <= hour < evening_end:
            return 'evening_peak'
        elif hour >= night_start or hour < night_end:
            return 'night'
        else:
            return 'normal'
    
    def _split_by_context(self, X, y):
        """
        Split data by context
        
        Args:
            X: Feature DataFrame
            y: Target values
            
        Returns:
            Dictionary mapping context to (X, y) tuples
        """
        context_data = {ctx: {'X': [], 'y': []} for ctx in self.contexts}
        
        for idx in range(len(X)):
            row = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]
            context = self._get_context(row)
            context_data[context]['X'].append(X.iloc[idx] if hasattr(X, 'iloc') else X[idx])
            context_data[context]['y'].append(y.iloc[idx] if hasattr(y, 'iloc') else y[idx])
        
        # Convert to DataFrames
        result = {}
        for ctx in self.contexts:
            if context_data[ctx]['X']:
                result[ctx] = (
                    pd.DataFrame(context_data[ctx]['X']),
                    pd.Series(context_data[ctx]['y'])
                )
        
        return result
    
    def fit(self, X, y, tune_hyperparameters=False):
        """
        Fit expert models for each context
        
        Args:
            X: Feature DataFrame (must include 'hour' and optionally 'is_raining')
            y: Target values
            tune_hyperparameters: Whether to tune each expert
            
        Returns:
            self
        """
        print("\n" + "=" * 60)
        print("TRAINING CONTEXT EXPERT ENSEMBLE")
        print("=" * 60)
        
        # Split data by context
        context_data = self._split_by_context(X, y)
        
        # Train expert for each context
        for context in self.contexts:
            if context not in context_data:
                print(f"\n[{context}] No data available, skipping...")
                continue
            
            X_ctx, y_ctx = context_data[context]
            n_samples = len(X_ctx)
            
            print(f"\n[{context}] Training expert with {n_samples} samples...")
            
            if n_samples < 10:
                print(f"    Insufficient data, will use default expert")
                continue
            
            # Create and train expert model
            expert = GradientBoostingModel(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1
            )
            
            if tune_hyperparameters and n_samples >= 50:
                # Reduced param grid for faster tuning
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'learning_rate': [0.05, 0.1]
                }
                expert.fit_with_tuning(X_ctx, y_ctx, param_grid=param_grid, cv=3)
            else:
                expert.fit(X_ctx, y_ctx)
            
            # Evaluate on training data
            metrics = expert.evaluate(X_ctx, y_ctx)
            print(f"    Training RMSE: {metrics['rmse']:.4f}")
            print(f"    Training R2: {metrics['r2']:.4f}")
            
            self.experts[context] = expert
        
        # Train default expert on all data
        print(f"\n[default] Training default expert with all {len(X)} samples...")
        self.default_expert = GradientBoostingModel(n_estimators=100, max_depth=5)
        self.default_expert.fit(X, y)
        
        self.is_fitted = True
        
        print("\n" + "-" * 60)
        print(f"Trained {len(self.experts)} context experts + 1 default expert")
        print("-" * 60)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using appropriate expert for each sample
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        predictions = np.zeros(len(X))
        
        for idx in range(len(X)):
            row = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]
            context = self._get_context(row)
            
            # Get appropriate expert
            if context in self.experts:
                expert = self.experts[context]
            else:
                expert = self.default_expert
            
            # Make prediction
            X_single = X.iloc[[idx]] if hasattr(X, 'iloc') else np.array([X[idx]])
            predictions[idx] = expert.predict(X_single)[0]
        
        return predictions
    
    def predict_with_context(self, X):
        """
        Make predictions and return context used
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Tuple of (predictions, contexts)
        """
        predictions = self.predict(X)
        contexts = [self._get_context(X.iloc[i] if hasattr(X, 'iloc') else X[i]) 
                   for i in range(len(X))]
        
        return predictions, contexts
    
    def evaluate(self, X, y):
        """
        Evaluate ensemble performance
        
        Args:
            X: Feature DataFrame
            y: True target values
            
        Returns:
            Dictionary with overall and per-context metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        predictions = self.predict(X)
        
        overall_metrics = {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions)
        }
        
        # Per-context metrics
        context_metrics = {}
        context_data = self._split_by_context(X, y)
        
        for context, (X_ctx, y_ctx) in context_data.items():
            if context in self.experts:
                pred_ctx = self.experts[context].predict(X_ctx)
            else:
                pred_ctx = self.default_expert.predict(X_ctx)
            
            context_metrics[context] = {
                'n_samples': len(y_ctx),
                'rmse': np.sqrt(mean_squared_error(y_ctx, pred_ctx)),
                'r2': r2_score(y_ctx, pred_ctx) if len(y_ctx) > 1 else 0
            }
        
        return {
            'overall': overall_metrics,
            'by_context': context_metrics
        }
    
    def get_expert_importance(self):
        """
        Get feature importance from each expert
        
        Returns:
            Dictionary mapping context to feature importance
        """
        importance = {}
        
        for context, expert in self.experts.items():
            importance[context] = expert.get_feature_importance()
        
        if self.default_expert:
            importance['default'] = self.default_expert.get_feature_importance()
        
        return importance
    
    def save(self, path=None):
        """Save ensemble to disk"""
        if path is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            path = os.path.join(MODELS_DIR, 'context_experts.joblib')
        
        joblib.dump({
            'experts': self.experts,
            'default_expert': self.default_expert,
            'contexts': self.contexts,
            'context_windows': self.context_windows
        }, path)
        
        print(f"Context Expert Ensemble saved to {path}")
        return path
    
    def load(self, path=None):
        """Load ensemble from disk"""
        if path is None:
            path = os.path.join(MODELS_DIR, 'context_experts.joblib')
        
        data = joblib.load(path)
        self.experts = data['experts']
        self.default_expert = data['default_expert']
        self.contexts = data['contexts']
        self.context_windows = data['context_windows']
        self.is_fitted = True
        
        print(f"Context Expert Ensemble loaded from {path}")
        return self


def test_context_experts():
    """Test context expert ensemble"""
    print("Testing Context Expert Ensemble...")
    print("=" * 60)
    
    # Generate sample data with context features
    np.random.seed(42)
    n_samples = 1000
    
    # Create features including hour and weather
    hours = np.random.randint(0, 24, n_samples)
    is_raining = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    X = pd.DataFrame({
        'hour': hours,
        'is_raining': is_raining,
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
        'temperature': np.random.uniform(0, 35, n_samples),
        'wind_speed': np.random.uniform(0, 50, n_samples)
    })
    
    # Create target with context-dependent patterns
    y = np.zeros(n_samples)
    for i in range(n_samples):
        base = X.iloc[i]['feature_1'] * 2 + X.iloc[i]['feature_2']
        
        # Add context-specific patterns
        if 7 <= hours[i] < 10:  # Morning peak
            y[i] = base + 3
        elif 16 <= hours[i] < 19:  # Evening peak
            y[i] = base + 4
        elif hours[i] >= 22 or hours[i] < 6:  # Night
            y[i] = base - 1
        else:
            y[i] = base
        
        # Rain effect
        if is_raining[i]:
            y[i] += 2
        
        # Add noise
        y[i] += np.random.randn() * 0.5
    
    y = pd.Series(y)
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train ensemble
    ensemble = ContextExpertEnsemble()
    ensemble.fit(X_train, y_train, tune_hyperparameters=False)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    metrics = ensemble.evaluate(X_test, y_test)
    
    print(f"\nOverall Performance:")
    print(f"  RMSE: {metrics['overall']['rmse']:.4f}")
    print(f"  R2: {metrics['overall']['r2']:.4f}")
    
    print(f"\nPer-Context Performance:")
    for context, ctx_metrics in metrics['by_context'].items():
        print(f"  {context}:")
        print(f"    Samples: {ctx_metrics['n_samples']}")
        print(f"    RMSE: {ctx_metrics['rmse']:.4f}")
    
    print("\n" + "=" * 60)
    print("Context Expert Ensemble Test Complete!")
    
    return ensemble


if __name__ == "__main__":
    test_context_experts()
