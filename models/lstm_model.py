"""
LSTM Model for Temporal Risk Prediction
Enhanced LSTM with attention mechanism for time series forecasting
"""
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODELS_DIR

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, Input, Attention,
        Concatenate, BatchNormalization, Bidirectional
    )
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. LSTM model will not work.")


class LSTMModel:
    """
    Enhanced LSTM Model for temporal risk prediction
    
    Features:
    - Multi-input LSTM architecture
    - Optional attention mechanism
    - Dropout for regularization
    - Early stopping
    - Bidirectional option
    """
    
    def __init__(self, sequence_length=24, n_features=10, 
                 lstm_units=64, dense_units=32, dropout_rate=0.2,
                 use_attention=True, bidirectional=False):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Number of time steps in input sequence
            n_features: Number of input features
            lstm_units: Number of LSTM units
            dense_units: Number of dense layer units
            dropout_rate: Dropout rate for regularization
            use_attention: Whether to use attention mechanism
            bidirectional: Whether to use bidirectional LSTM
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        
        self.model = None
        self.history = None
        self.is_fitted = False
        self.scaler_X = None
        self.scaler_y = None
        
        if TF_AVAILABLE:
            self._build_model()
    
    def _build_model(self):
        """Build the LSTM model architecture"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # First LSTM layer
        if self.bidirectional:
            x = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(inputs)
        else:
            x = LSTM(self.lstm_units, return_sequences=True)(inputs)
        
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Second LSTM layer
        if self.use_attention:
            # LSTM with return sequences for attention
            if self.bidirectional:
                lstm_out = Bidirectional(LSTM(self.lstm_units // 2, return_sequences=True))(x)
            else:
                lstm_out = LSTM(self.lstm_units // 2, return_sequences=True)(x)
            
            # Simple attention mechanism
            attention = Dense(1, activation='tanh')(lstm_out)
            attention = tf.keras.layers.Flatten()(attention)
            attention = tf.keras.layers.Activation('softmax')(attention)
            attention = tf.keras.layers.RepeatVector(self.lstm_units // 2 * (2 if self.bidirectional else 1))(attention)
            attention = tf.keras.layers.Permute([2, 1])(attention)
            
            # Apply attention
            attended = tf.keras.layers.Multiply()([lstm_out, attention])
            x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(attended)
        else:
            # Standard LSTM without attention
            if self.bidirectional:
                x = Bidirectional(LSTM(self.lstm_units // 2))(x)
            else:
                x = LSTM(self.lstm_units // 2)(x)
        
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Dense layers
        x = Dense(self.dense_units, activation='relu')(x)
        x = Dropout(self.dropout_rate / 2)(x)
        
        # Output layer
        outputs = Dense(1, activation='linear')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    
    def create_sequences(self, X, y=None):
        """
        Create sequences for LSTM input
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (optional)
            
        Returns:
            X_seq: Sequences (n_sequences, sequence_length, n_features)
            y_seq: Target values for each sequence (if y provided)
        """
        X_array = np.array(X)
        n_samples = len(X_array)
        
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(n_samples - self.sequence_length):
            X_seq.append(X_array[i:i + self.sequence_length])
            if y is not None:
                y_seq.append(y.iloc[i + self.sequence_length] if hasattr(y, 'iloc') else y[i + self.sequence_length])
        
        X_seq = np.array(X_seq)
        
        if y_seq is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        
        return X_seq
    
    def fit(self, X, y, validation_split=0.2, epochs=100, batch_size=32, verbose=1):
        """
        Fit the LSTM model
        
        Args:
            X: Feature matrix
            y: Target values
            validation_split: Fraction of data for validation
            epochs: Maximum number of epochs
            batch_size: Training batch size
            verbose: Verbosity level
            
        Returns:
            self
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)
        
        # Update n_features if needed
        if X_seq.shape[2] != self.n_features:
            self.n_features = X_seq.shape[2]
            self._build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=verbose
            )
        ]
        
        # Fit model
        self.history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Feature matrix (will be converted to sequences)
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_seq = self.create_sequences(X)
        return self.model.predict(X_seq, verbose=0).flatten()
    
    def predict_next(self, X_sequence):
        """
        Predict next value given a sequence
        
        Args:
            X_sequence: Single sequence (sequence_length, n_features)
            
        Returns:
            Predicted value
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_input = np.array(X_sequence).reshape(1, self.sequence_length, self.n_features)
        return self.model.predict(X_input, verbose=0)[0, 0]
    
    def evaluate(self, X, y):
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with metrics
        """
        X_seq, y_seq = self.create_sequences(X, y)
        
        loss, mae = self.model.evaluate(X_seq, y_seq, verbose=0)
        y_pred = self.model.predict(X_seq, verbose=0).flatten()
        
        from sklearn.metrics import mean_squared_error, r2_score
        
        return {
            'loss': loss,
            'mse': mean_squared_error(y_seq, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_seq, y_pred)),
            'mae': mae,
            'r2': r2_score(y_seq, y_pred)
        }
    
    def save(self, path=None):
        """Save model to disk"""
        if path is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            path = os.path.join(MODELS_DIR, 'lstm_model')
        
        self.model.save(path)
        print(f"LSTM model saved to {path}")
        return path
    
    def load(self, path=None):
        """Load model from disk"""
        if path is None:
            path = os.path.join(MODELS_DIR, 'lstm_model')
        
        self.model = keras.models.load_model(path)
        self.is_fitted = True
        print(f"LSTM model loaded from {path}")
        return self
    
    def get_training_history(self):
        """Get training history"""
        if self.history is None:
            return {}
        
        return {
            'loss': self.history.history['loss'],
            'val_loss': self.history.history.get('val_loss', []),
            'mae': self.history.history.get('mae', []),
            'val_mae': self.history.history.get('val_mae', [])
        }


class SimpleLSTMModel:
    """
    Simplified LSTM model for when TensorFlow is not available
    Uses a basic numpy-based approach
    """
    
    def __init__(self, sequence_length=24):
        self.sequence_length = sequence_length
        self.weights = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """Simple weighted average approach"""
        # Use exponential weights for recent values
        self.weights = np.exp(np.linspace(-1, 0, self.sequence_length))
        self.weights /= self.weights.sum()
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict using weighted average"""
        X_array = np.array(X)
        predictions = []
        
        for i in range(len(X_array) - self.sequence_length):
            seq = X_array[i:i + self.sequence_length, 0]  # Use first feature
            pred = np.dot(seq, self.weights)
            predictions.append(pred)
        
        return np.array(predictions)


def test_lstm():
    """Test LSTM model"""
    print("Testing LSTM Model...")
    print("=" * 60)
    
    if not TF_AVAILABLE:
        print("TensorFlow not available. Using SimpleLSTMModel.")
        model = SimpleLSTMModel(sequence_length=10)
    else:
        print("TensorFlow available. Using full LSTMModel.")
        
        # Generate sample time series data
        np.random.seed(42)
        n_samples = 500
        n_features = 5
        
        # Create synthetic time series
        t = np.linspace(0, 50, n_samples)
        base_signal = np.sin(t) + 0.5 * np.sin(2 * t)
        noise = np.random.randn(n_samples) * 0.2
        y = base_signal + noise
        
        # Create features
        X = pd.DataFrame({
            f'feature_{i}': np.roll(y, i) + np.random.randn(n_samples) * 0.1
            for i in range(n_features)
        })
        y = pd.Series(y)
        
        # Split data
        train_size = int(0.8 * n_samples)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create and train model
        model = LSTMModel(
            sequence_length=24,
            n_features=n_features,
            lstm_units=32,
            use_attention=True,
            bidirectional=False
        )
        
        print("\nTraining LSTM model...")
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
        
        # Evaluate
        print("\nEvaluating model...")
        metrics = model.evaluate(X_test, y_test)
        print(f"Test RMSE: {metrics['rmse']:.4f}")
        print(f"Test R2: {metrics['r2']:.4f}")
    
    print("\n" + "=" * 60)
    print("LSTM Model Test Complete!")
    
    return model


if __name__ == "__main__":
    test_lstm()
