"""
Model Trainer - Unified Training Pipeline
Orchestrates training of all models with advanced fine-tuning
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODELS_DIR
from models.base_models import LinearRegressionModel, GradientBoostingModel
from models.context_experts import ContextExpertEnsemble
from models.online_learner import OnlineLearner


class ModelTrainer:
    """
    Unified Model Training Pipeline
    
    Trains and manages:
    - Linear Regression (baseline)
    - Gradient Boosting (primary)
    - Context Expert Ensemble (novel)
    - Online Learner (continuous improvement)
    """
    
    def __init__(self):
        self.models = {}
        self.training_history = []
        self.best_model = None
        self.best_metrics = None
        
    def train_all_models(self, X, y, test_size=0.2, tune_hyperparameters=True):
        """
        Train all models and select best performer
        
        Args:
            X: Feature DataFrame
            y: Target values
            test_size: Fraction for test set
            tune_hyperparameters: Whether to tune hyperparameters
            
        Returns:
            Dictionary with all trained models and metrics
        """
        print("\n" + "=" * 70)
        print("URBAN PULSE - MODEL TRAINING PIPELINE")
        print("=" * 70)
        
        start_time = datetime.now()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"\nData Split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Testing: {len(X_test)} samples")
        
        results = {}
        
        # 1. Linear Regression (Baseline)
        print("\n" + "-" * 70)
        print("1. TRAINING LINEAR REGRESSION (Baseline)")
        print("-" * 70)
        
        lr_model = LinearRegressionModel(regularization='ridge')
        lr_model.fit(X_train, y_train)
        lr_metrics = lr_model.evaluate(X_test, y_test)
        
        self.models['linear_regression'] = lr_model
        results['linear_regression'] = lr_metrics
        
        print(f"   Test RMSE: {lr_metrics['rmse']:.4f}")
        print(f"   Test R2: {lr_metrics['r2']:.4f}")
        
        # 2. Gradient Boosting (Primary)
        print("\n" + "-" * 70)
        print("2. TRAINING GRADIENT BOOSTING (Primary)")
        print("-" * 70)
        
        gb_model = GradientBoostingModel()
        
        if tune_hyperparameters:
            print("   Running hyperparameter tuning...")
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.15]
            }
            gb_model.fit_with_tuning(X_train, y_train, param_grid=param_grid, cv=3)
        else:
            gb_model.fit(X_train, y_train)
        
        gb_metrics = gb_model.evaluate(X_test, y_test)
        
        self.models['gradient_boosting'] = gb_model
        results['gradient_boosting'] = gb_metrics
        
        print(f"   Test RMSE: {gb_metrics['rmse']:.4f}")
        print(f"   Test R2: {gb_metrics['r2']:.4f}")
        
        # 3. Context Expert Ensemble (Novel)
        print("\n" + "-" * 70)
        print("3. TRAINING CONTEXT EXPERT ENSEMBLE (Novel)")
        print("-" * 70)
        
        # Check if context features are available
        if 'hour' in X.columns:
            ce_model = ContextExpertEnsemble()
            ce_model.fit(X_train, y_train, tune_hyperparameters=False)
            ce_metrics = ce_model.evaluate(X_test, y_test)
            
            self.models['context_experts'] = ce_model
            results['context_experts'] = ce_metrics['overall']
            
            print(f"   Test RMSE: {ce_metrics['overall']['rmse']:.4f}")
            print(f"   Test R2: {ce_metrics['overall']['r2']:.4f}")
        else:
            print("   Skipped - 'hour' column not available")
        
        # 4. Online Learner (Continuous Improvement)
        print("\n" + "-" * 70)
        print("4. INITIALIZING ONLINE LEARNER")
        print("-" * 70)
        
        ol_model = OnlineLearner()
        ol_model.initialize(X_train, y_train)
        ol_metrics = ol_model.evaluate(X_test, y_test)
        
        self.models['online_learner'] = ol_model
        results['online_learner'] = ol_metrics
        
        print(f"   Test RMSE: {ol_metrics['rmse']:.4f}")
        print(f"   Test R2: {ol_metrics['r2']:.4f}")
        
        # Select best model
        print("\n" + "-" * 70)
        print("MODEL COMPARISON")
        print("-" * 70)
        
        best_name = None
        best_rmse = float('inf')
        
        for name, metrics in results.items():
            rmse = metrics.get('rmse', float('inf'))
            r2 = metrics.get('r2', 0)
            marker = ""
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name
            
            print(f"   {name}: RMSE={rmse:.4f}, R2={r2:.4f}")
        
        self.best_model = best_name
        self.best_metrics = results[best_name]
        
        print(f"\n   Best Model: {best_name}")
        
        # Record training
        elapsed = datetime.now() - start_time
        self.training_history.append({
            'timestamp': datetime.now(),
            'n_samples': len(X),
            'results': results,
            'best_model': best_name,
            'elapsed': str(elapsed)
        })
        
        print("\n" + "=" * 70)
        print(f"TRAINING COMPLETE - Elapsed: {elapsed}")
        print("=" * 70)
        
        return results
    
    def get_best_model(self):
        """Get the best performing model"""
        if self.best_model is None:
            raise ValueError("No models trained yet")
        return self.models[self.best_model]
    
    def predict(self, X, model_name=None):
        """
        Make predictions using specified or best model
        
        Args:
            X: Feature DataFrame
            model_name: Model to use (uses best if None)
            
        Returns:
            Predictions array
        """
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].predict(X)
    
    def update_online_model(self, X_new, y_new):
        """
        Update online learner with new data
        
        Args:
            X_new: New feature data
            y_new: New target values
            
        Returns:
            Updated metrics
        """
        if 'online_learner' not in self.models:
            raise ValueError("Online learner not initialized")
        
        self.models['online_learner'].partial_fit(X_new, y_new)
        return self.models['online_learner'].evaluate(X_new, y_new)
    
    def get_feature_importance(self, model_name=None):
        """Get feature importance from specified model"""
        if model_name is None:
            model_name = self.best_model
        
        model = self.models.get(model_name)
        
        if hasattr(model, 'get_feature_importance'):
            return model.get_feature_importance()
        elif hasattr(model, 'get_coefficients'):
            return model.get_coefficients()
        
        return {}
    
    def save_all_models(self, directory=None):
        """Save all trained models"""
        if directory is None:
            directory = MODELS_DIR
        
        os.makedirs(directory, exist_ok=True)
        
        saved_paths = {}
        for name, model in self.models.items():
            if hasattr(model, 'save'):
                path = model.save(os.path.join(directory, f"{name}.joblib"))
                saved_paths[name] = path
        
        print(f"Saved {len(saved_paths)} models to {directory}")
        return saved_paths
    
    def load_model(self, model_name, path=None):
        """Load a specific model"""
        if path is None:
            path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
        
        if model_name == 'linear_regression':
            model = LinearRegressionModel()
        elif model_name == 'gradient_boosting':
            model = GradientBoostingModel()
        elif model_name == 'context_experts':
            model = ContextExpertEnsemble()
        elif model_name == 'online_learner':
            model = OnlineLearner()
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        model.load(path)
        self.models[model_name] = model
        
        return model
    
    def cross_validate(self, X, y, model_name='gradient_boosting', n_splits=5):
        """
        Perform time series cross-validation
        
        Args:
            X: Feature DataFrame
            y: Target values
            model_name: Model to validate
            n_splits: Number of CV splits
            
        Returns:
            Dictionary with CV results
        """
        from sklearn.metrics import mean_squared_error, r2_score
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        rmse_scores = []
        r2_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create fresh model
            if model_name == 'gradient_boosting':
                model = GradientBoostingModel()
            else:
                model = LinearRegressionModel()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            rmse_scores.append(rmse)
            r2_scores.append(r2)
        
        return {
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'fold_scores': list(zip(rmse_scores, r2_scores))
        }


def main():
    """Test model trainer"""
    print("Testing Model Trainer...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 500
    
    X = pd.DataFrame({
        'hour': np.random.randint(0, 24, n_samples),
        'is_raining': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
        'temperature': np.random.uniform(0, 35, n_samples),
        'wind_speed': np.random.uniform(0, 50, n_samples)
    })
    
    y = (2 * X['feature_1'] + X['feature_2'] + 
         0.1 * X['hour'] + 
         2 * X['is_raining'] + 
         np.random.randn(n_samples) * 0.5)
    
    # Train all models
    trainer = ModelTrainer()
    results = trainer.train_all_models(X, y, tune_hyperparameters=False)
    
    # Get feature importance
    print("\nTop Features (Gradient Boosting):")
    importance = trainer.get_feature_importance('gradient_boosting')
    for feat, imp in list(importance.items())[:5]:
        print(f"  {feat}: {imp:.4f}")
    
    return trainer


if __name__ == "__main__":
    main()
