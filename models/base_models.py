"""
Base ML Models for Risk Prediction
Linear Regression (baseline) and Gradient Boosting (primary)
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODELS_DIR


class BaseModel:
    """Base class for all models"""
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.metrics = {}
        
    def fit(self, X, y):
        raise NotImplementedError
        
    def predict(self, X):
        raise NotImplementedError
        
    def evaluate(self, X, y):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        
        self.metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        return self.metrics
    
    def save(self, path=None):
        """Save model to disk"""
        if path is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            path = os.path.join(MODELS_DIR, f"{self.name}.joblib")
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }, path)
        
        print(f"Model saved to {path}")
        return path
    
    def load(self, path=None):
        """Load model from disk"""
        if path is None:
            path = os.path.join(MODELS_DIR, f"{self.name}.joblib")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.metrics = data.get('metrics', {})
        self.is_fitted = True
        
        print(f"Model loaded from {path}")
        return self


class LinearRegressionModel(BaseModel):
    """
    Linear Regression Model (Baseline)
    Simple, interpretable model for baseline comparison
    """
    
    def __init__(self, regularization='ridge', alpha=1.0):
        super().__init__('linear_regression')
        self.regularization = regularization
        self.alpha = alpha
        
        if regularization == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            self.model = Lasso(alpha=alpha)
        else:
            self.model = LinearRegression()
    
    def fit(self, X, y, scale=True):
        """
        Fit the linear regression model
        
        Args:
            X: Feature matrix
            y: Target values
            scale: Whether to scale features
            
        Returns:
            self
        """
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Handle NaN values
        X_array = np.nan_to_num(X_array, nan=0)
        y_array = np.nan_to_num(y_array, nan=0)
        
        if scale:
            X_scaled = self.scaler.fit_transform(X_array)
        else:
            X_scaled = X_array
        
        self.model.fit(X_scaled, y_array)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_array = np.array(X)
        X_array = np.nan_to_num(X_array, nan=0)
        X_scaled = self.scaler.transform(X_array)
        
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self):
        """Get feature coefficients"""
        if not self.is_fitted:
            return {}
        
        coefs = self.model.coef_
        
        if self.feature_names:
            return dict(zip(self.feature_names, coefs))
        return coefs


class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting Model (Primary)
    Powerful ensemble model for risk prediction
    """
    
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1):
        super().__init__('gradient_boosting')
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        self.best_params = None
    
    def fit(self, X, y, scale=False):
        """
        Fit the gradient boosting model
        
        Args:
            X: Feature matrix
            y: Target values
            scale: Whether to scale features (not required for tree models)
            
        Returns:
            self
        """
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Handle NaN values
        X_array = np.nan_to_num(X_array, nan=0)
        y_array = np.nan_to_num(y_array, nan=0)
        
        if scale:
            X_array = self.scaler.fit_transform(X_array)
        
        self.model.fit(X_array, y_array)
        self.is_fitted = True
        
        return self
    
    def fit_with_tuning(self, X, y, param_grid=None, cv=5):
        """
        Fit with hyperparameter tuning using GridSearchCV
        
        Args:
            X: Feature matrix
            y: Target values
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds
            
        Returns:
            self
        """
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        X_array = np.nan_to_num(X_array, nan=0)
        y_array = np.nan_to_num(y_array, nan=0)
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2],
                'min_samples_split': [2, 5, 10]
            }
        
        # Use TimeSeriesSplit for temporal data
        tscv = TimeSeriesSplit(n_splits=cv)
        
        grid_search = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        print("Running hyperparameter tuning...")
        grid_search.fit(X_array, y_array)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.is_fitted = True
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {-grid_search.best_score_:.4f} MSE")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_array = np.array(X)
        X_array = np.nan_to_num(X_array, nan=0)
        
        return self.model.predict(X_array)
    
    def get_feature_importance(self):
        """Get feature importances"""
        if not self.is_fitted:
            return {}
        
        importances = self.model.feature_importances_
        
        if self.feature_names:
            importance_dict = dict(zip(self.feature_names, importances))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return importances
    
    def get_top_features(self, n=10):
        """Get top N most important features"""
        importance = self.get_feature_importance()
        if isinstance(importance, dict):
            return dict(list(importance.items())[:n])
        return importance[:n]


def test_models():
    """Test base models"""
    print("Testing Base Models...")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = 3 * X['feature_0'] + 2 * X['feature_1'] - X['feature_2'] + np.random.randn(n_samples) * 0.5
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Test Linear Regression
    print("\n1. Linear Regression Model")
    print("-" * 40)
    lr_model = LinearRegressionModel(regularization='ridge')
    lr_model.fit(X_train, y_train)
    lr_metrics = lr_model.evaluate(X_test, y_test)
    print(f"   RMSE: {lr_metrics['rmse']:.4f}")
    print(f"   R2: {lr_metrics['r2']:.4f}")
    
    # Test Gradient Boosting
    print("\n2. Gradient Boosting Model")
    print("-" * 40)
    gb_model = GradientBoostingModel(n_estimators=50, max_depth=3)
    gb_model.fit(X_train, y_train)
    gb_metrics = gb_model.evaluate(X_test, y_test)
    print(f"   RMSE: {gb_metrics['rmse']:.4f}")
    print(f"   R2: {gb_metrics['r2']:.4f}")
    
    # Feature importance
    print("\n   Top 5 Features:")
    top_features = gb_model.get_top_features(5)
    for feat, imp in top_features.items():
        print(f"     {feat}: {imp:.4f}")
    
    print("\n" + "=" * 60)
    print("Base Models Test Complete!")
    
    return lr_model, gb_model


if __name__ == "__main__":
    test_models()
