import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

class AirQualityModel:
    """
    Advanced multi-output air quality prediction models for SIH 2025
    Supports multiple model types: LightGBM, XGBoost, Random Forest, LSTM
    """

    def __init__(self, model_type='lightgbm', random_state=42):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model = None
        self.is_fitted = False

    def _create_model(self, n_features):
        """Create model based on type"""
        if self.model_type == 'lightgbm':
            base_model = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=self.random_state,
                verbose=-1
            )
            return MultiOutputRegressor(base_model)

        elif self.model_type == 'xgboost':
            base_model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=0
            )
            return MultiOutputRegressor(base_model)

        elif self.model_type == 'randomforest':
            base_model = RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
            return MultiOutputRegressor(base_model)

        elif self.model_type == 'gradientboosting':
            base_model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=self.random_state
            )
            return MultiOutputRegressor(base_model)

        else:  # Default to LightGBM
            base_model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.1,
                num_leaves=31,
                random_state=self.random_state,
                verbose=-1
            )
            return MultiOutputRegressor(base_model)

    def train(self, X, y):
        """
        Train the multi-output model

        Args:
            X: Features DataFrame
            y: Targets DataFrame with columns ['O3_target', 'NO2_target']
        """
        # Ensure y has correct columns
        if isinstance(y, pd.DataFrame):
            if 'O3_target' in y.columns and 'NO2_target' in y.columns:
                y = y[['O3_target', 'NO2_target']].values
            else:
                y = y.values

        # Remove any rows with NaN targets
        mask = ~np.isnan(y).any(axis=1)
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) == 0:
            raise ValueError("No valid training data after removing NaN targets")

        # Create and train model
        self.model = self._create_model(X_clean.shape[1])

        print(f"Training {self.model_type} model with {len(X_clean)} samples...")
        self.model.fit(X_clean, y_clean)
        self.is_fitted = True

        # Calculate training metrics
        train_pred = self.model.predict(X_clean)
        metrics = self._calculate_metrics(y_clean, train_pred)

        print("Training completed!")
        print(f"O3 - RMSE: {metrics['o3_rmse']:.2f}, R²: {metrics['o3_r2']:.3f}")
        print(f"NO2 - RMSE: {metrics['no2_rmse']:.2f}, R²: {metrics['no2_r2']:.3f}")

        return metrics

    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            # Return dummy predictions for demo
            print("Model not fitted. Returning demo predictions...")
            n_samples = len(X)

            # Generate realistic demo predictions based on input forecasts
            if 'O3_forecast' in X.columns and 'NO2_forecast' in X.columns:
                o3_base = X['O3_forecast'].values
                no2_base = X['NO2_forecast'].values
            else:
                o3_base = np.full(n_samples, 35.0)
                no2_base = np.full(n_samples, 45.0)

            # Add some realistic noise and adjustments
            o3_pred = o3_base + np.random.normal(0, 3, n_samples)
            no2_pred = no2_base + np.random.normal(0, 5, n_samples)

            return np.column_stack([o3_pred, no2_pred])

        return self.model.predict(X)

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics"""
        o3_rmse = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
        no2_rmse = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))

        o3_r2 = r2_score(y_true[:, 0], y_pred[:, 0])
        no2_r2 = r2_score(y_true[:, 1], y_pred[:, 1])

        o3_mae = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
        no2_mae = mean_absolute_error(y_true[:, 1], y_pred[:, 1])

        # Refined Index of Agreement (RIA)
        o3_ria = self._calculate_ria(y_true[:, 0], y_pred[:, 0])
        no2_ria = self._calculate_ria(y_true[:, 1], y_pred[:, 1])

        return {
            'o3_rmse': o3_rmse, 'o3_r2': o3_r2, 'o3_mae': o3_mae, 'o3_ria': o3_ria,
            'no2_rmse': no2_rmse, 'no2_r2': no2_r2, 'no2_mae': no2_mae, 'no2_ria': no2_ria
        }

    def _calculate_ria(self, y_true, y_pred):
        """Calculate Refined Index of Agreement"""
        mean_obs = np.mean(y_true)
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((np.abs(y_pred - mean_obs) + np.abs(y_true - mean_obs)) ** 2)
        return 1 - (numerator / denominator) if denominator != 0 else 0

    def save_model(self, filepath):
        """Save trained model"""
        if self.is_fitted:
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'is_fitted': self.is_fitted
            }
            joblib.dump(model_data, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("Cannot save unfitted model")

    def load_model(self, filepath):
        """Load trained model"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.is_fitted = model_data['is_fitted']
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_fitted = False

class LSTMModel:
    """
    LSTM-based model for sequence prediction
    For advanced temporal modeling of air quality data
    """

    def __init__(self, sequence_length=24, lstm_units=50, dense_units=25):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.model = None
        self.scaler = None

    def create_sequences(self, data, target_cols):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            if target_cols is not None:
                y.append(data[i + self.sequence_length][target_cols])
        return np.array(X), np.array(y)

    def build_model(self, input_shape, output_dim=2):
        """Build LSTM model architecture"""
        self.model = Sequential([
            Input(shape=input_shape),
            LSTM(self.lstm_units, return_sequences=True, dropout=0.2),
            LSTM(self.lstm_units//2, dropout=0.2),
            Dense(self.dense_units, activation='relu'),
            Dropout(0.3),
            Dense(output_dim, activation='linear')
        ])

        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        return self.model

    def train(self, X, y, validation_split=0.2, epochs=100):
        """Train LSTM model"""
        if self.model is None:
            self.build_model((X.shape[1], X.shape[2]))

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        return history

    def predict(self, X):
        """Make predictions with LSTM model"""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)
