"""
Online Learning Module
Enables continuous model improvement with new data batches
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODELS_DIR, FEEDBACK_LEARNING_RATE


class OnlineLearner:
    """
    Online Learning Module for Continuous Model Improvement
    
    Features:
    - Incremental learning with new data batches
    - Model parameter updates without full retraining
    - Performance tracking over time
    - Automatic model versioning
    """
    
    def __init__(self, learning_rate=FEEDBACK_LEARNING_RATE):
        self.learning_rate = learning_rate
        self.model = SGDRegressor(
            loss='squared_error',
            penalty='l2',
            alpha=0.0001,
            learning_rate='adaptive',
            eta0=learning_rate,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_initialized = False
        self.update_history = []
        self.feature_names = None
        self.version = 0
        
    def initialize(self, X, y):
        """
        Initialize the online learner with initial data
        
        Args:
            X: Initial feature matrix
            y: Initial target values
            
        Returns:
            self
        """
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Handle NaN
        X_array = np.nan_to_num(X_array, nan=0)
        y_array = np.nan_to_num(y_array, nan=0)
        
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Initial fit
        self.model.fit(X_scaled, y_array)
        self.is_initialized = True
        self.version = 1
        
        # Record initialization
        self._record_update(len(X), 'initialization')
        
        print(f"Online Learner initialized with {len(X)} samples")
        return self
    
    def partial_fit(self, X, y, sample_weight=None):
        """
        Update model with new data batch (incremental learning)
        
        Args:
            X: New feature matrix
            y: New target values
            sample_weight: Optional sample weights
            
        Returns:
            self
        """
        if not self.is_initialized:
            return self.initialize(X, y)
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Handle NaN
        X_array = np.nan_to_num(X_array, nan=0)
        y_array = np.nan_to_num(y_array, nan=0)
        
        # Scale using existing scaler
        X_scaled = self.scaler.transform(X_array)
        
        # Partial fit
        self.model.partial_fit(X_scaled, y_array, sample_weight=sample_weight)
        self.version += 1
        
        # Record update
        self._record_update(len(X), 'partial_fit')
        
        return self
    
    def update_with_feedback(self, X, y_true, y_pred):
        """
        Update model based on prediction feedback
        
        Args:
            X: Features for predictions
            y_true: Actual observed values
            y_pred: Model's predictions
            
        Returns:
            Dictionary with update metrics
        """
        # Calculate prediction errors
        errors = y_true - y_pred
        
        # Weight samples by error magnitude (focus on larger errors)
        weights = np.abs(errors) / (np.abs(errors).max() + 1e-6)
        weights = weights / weights.sum() * len(weights)  # Normalize
        
        # Update model with weighted samples
        self.partial_fit(X, y_true, sample_weight=weights)
        
        # Calculate improvement metrics
        metrics = {
            'mean_error': np.mean(errors),
            'mean_abs_error': np.mean(np.abs(errors)),
            'max_error': np.max(np.abs(errors)),
            'samples_updated': len(X),
            'version': self.version
        }
        
        return metrics
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_initialized:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        X_array = np.array(X)
        X_array = np.nan_to_num(X_array, nan=0)
        X_scaled = self.scaler.transform(X_array)
        
        return self.model.predict(X_scaled)
    
    def evaluate(self, X, y):
        """Evaluate current model performance"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_pred = self.predict(X)
        
        return {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'version': self.version
        }
    
    def _record_update(self, n_samples, update_type):
        """Record update in history"""
        self.update_history.append({
            'timestamp': datetime.now(),
            'type': update_type,
            'n_samples': n_samples,
            'version': self.version
        })
    
    def get_update_history(self):
        """Get model update history"""
        return pd.DataFrame(self.update_history)
    
    def get_coefficients(self):
        """Get current model coefficients"""
        if not self.is_initialized:
            return {}
        
        coefs = self.model.coef_
        
        if self.feature_names:
            return dict(zip(self.feature_names, coefs))
        return coefs
    
    def save(self, path=None):
        """Save online learner state"""
        if path is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            path = os.path.join(MODELS_DIR, f'online_learner_v{self.version}.joblib')
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'version': self.version,
            'update_history': self.update_history,
            'is_initialized': self.is_initialized
        }, path)
        
        print(f"Online Learner v{self.version} saved to {path}")
        return path
    
    def load(self, path=None):
        """Load online learner state"""
        if path is None:
            # Find latest version
            if os.path.exists(MODELS_DIR):
                files = [f for f in os.listdir(MODELS_DIR) if f.startswith('online_learner_v')]
                if files:
                    path = os.path.join(MODELS_DIR, max(files))
        
        if path is None or not os.path.exists(path):
            raise FileNotFoundError("No saved online learner found")
        
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.version = data['version']
        self.update_history = data['update_history']
        self.is_initialized = data['is_initialized']
        
        print(f"Online Learner v{self.version} loaded from {path}")
        return self


class AdaptiveModelManager:
    """
    Manages multiple models and selects best performer
    Enables A/B testing of model updates
    """
    
    def __init__(self):
        self.models = {}
        self.performance_history = []
        self.active_model = None
        
    def register_model(self, name, model):
        """Register a model for management"""
        self.models[name] = {
            'model': model,
            'metrics': {},
            'is_active': False
        }
        
        if self.active_model is None:
            self.set_active(name)
    
    def set_active(self, name):
        """Set the active model for predictions"""
        if name not in self.models:
            raise ValueError(f"Model {name} not registered")
        
        # Deactivate current
        if self.active_model:
            self.models[self.active_model]['is_active'] = False
        
        # Activate new
        self.models[name]['is_active'] = True
        self.active_model = name
        
        print(f"Active model set to: {name}")
    
    def predict(self, X):
        """Make predictions using active model"""
        if self.active_model is None:
            raise ValueError("No active model set")
        
        model = self.models[self.active_model]['model']
        return model.predict(X)
    
    def evaluate_all(self, X, y):
        """Evaluate all registered models"""
        results = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            if hasattr(model, 'evaluate'):
                metrics = model.evaluate(X, y)
            else:
                y_pred = model.predict(X)
                from sklearn.metrics import mean_squared_error, r2_score
                metrics = {
                    'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                    'r2': r2_score(y, y_pred)
                }
            
            results[name] = metrics
            self.models[name]['metrics'] = metrics
        
        # Record in history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'results': results
        })
        
        return results
    
    def select_best(self, X, y, metric='rmse'):
        """
        Evaluate all models and select best performer
        
        Args:
            X: Evaluation features
            y: Evaluation targets
            metric: Metric to use for selection (lower is better for rmse/mae)
            
        Returns:
            Name of best model
        """
        results = self.evaluate_all(X, y)
        
        # Find best (assuming lower is better)
        best_name = min(results.keys(), key=lambda k: results[k].get(metric, float('inf')))
        
        print(f"\nModel Comparison ({metric}):")
        for name, metrics in results.items():
            marker = " *" if name == best_name else ""
            print(f"  {name}: {metrics.get(metric, 'N/A'):.4f}{marker}")
        
        self.set_active(best_name)
        return best_name


def test_online_learner():
    """Test online learning functionality"""
    print("Testing Online Learner...")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate initial data
    n_initial = 200
    n_features = 5
    
    X_initial = pd.DataFrame(
        np.random.randn(n_initial, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_initial = 2 * X_initial['feature_0'] + X_initial['feature_1'] + np.random.randn(n_initial) * 0.3
    
    # Initialize learner
    learner = OnlineLearner()
    learner.initialize(X_initial, y_initial)
    
    # Evaluate initial performance
    metrics = learner.evaluate(X_initial, y_initial)
    print(f"\nInitial Performance:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R2: {metrics['r2']:.4f}")
    
    # Simulate online updates
    print("\nSimulating online updates...")
    for batch in range(5):
        # Generate new batch
        n_batch = 50
        X_batch = pd.DataFrame(
            np.random.randn(n_batch, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y_batch = 2 * X_batch['feature_0'] + X_batch['feature_1'] + np.random.randn(n_batch) * 0.3
        
        # Update model
        learner.partial_fit(X_batch, y_batch)
        
        # Evaluate
        metrics = learner.evaluate(X_batch, y_batch)
        print(f"  Batch {batch + 1}: RMSE={metrics['rmse']:.4f}, Version={metrics['version']}")
    
    # Show update history
    print("\nUpdate History:")
    history = learner.get_update_history()
    print(history)
    
    print("\n" + "=" * 60)
    print("Online Learner Test Complete!")
    
    return learner


if __name__ == "__main__":
    test_online_learner()
