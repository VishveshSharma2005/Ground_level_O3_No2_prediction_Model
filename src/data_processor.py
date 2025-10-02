import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

class DataProcessor:
    """
    Comprehensive data processing for SIH 2025 Air Quality Forecasting
    Handles satellite data interpolation, feature engineering, and preprocessing
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None

    def engineer_features(self, df):
        """
        Advanced feature engineering for air quality prediction

        Args:
            df: DataFrame with columns [year, month, day, hour, O3_forecast, NO2_forecast, 
                T_forecast, q_forecast, u_forecast, v_forecast, w_forecast, 
                NO2_satellite, HCHO_satellite, ratio_satellite]

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Convert to datetime
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

        # Handle satellite data (daily to hourly interpolation)
        df = self._interpolate_satellite_data(df)

        # Temporal features
        df = self._create_temporal_features(df)

        # Meteorological features
        df = self._create_meteorological_features(df)

        # Lag features
        df = self._create_lag_features(df)

        # Rolling statistics
        df = self._create_rolling_features(df)

        # Interaction features
        df = self._create_interaction_features(df)

        # Clean up
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        return df

    def _interpolate_satellite_data(self, df):
        """Interpolate sparse satellite data from daily to hourly"""
        satellite_cols = ['NO2_satellite', 'HCHO_satellite', 'ratio_satellite']

        for col in satellite_cols:
            if col in df.columns:
                # Group by date and forward-fill within each day
                df['date'] = df['datetime'].dt.date
                df[col] = df.groupby('date')[col].transform('ffill')
                df[col] = df.groupby('date')[col].transform('bfill')

                # If still NaN, interpolate linearly
                df[col] = df[col].interpolate(method='linear')

        df = df.drop('date', axis=1, errors='ignore')
        return df

    def _create_temporal_features(self, df):
        """Create cyclical temporal features"""
        # Hour cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Month cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Season encoding
        df['season'] = df['month'].apply(self._get_season)

        # Rush hour indicators
        df['morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)

        return df

    def _create_meteorological_features(self, df):
        """Create advanced meteorological features"""
        # Wind speed and direction
        if 'u_forecast' in df.columns and 'v_forecast' in df.columns:
            df['wind_speed'] = np.sqrt(df['u_forecast']**2 + df['v_forecast']**2)
            df['wind_direction'] = np.arctan2(df['v_forecast'], df['u_forecast'])
            df['wind_direction_sin'] = np.sin(df['wind_direction'])
            df['wind_direction_cos'] = np.cos(df['wind_direction'])

        # Temperature features
        if 'T_forecast' in df.columns:
            df['temp_anomaly'] = df['T_forecast'] - df['T_forecast'].rolling(24).mean()
            df['temp_gradient'] = df['T_forecast'].diff()

        # Humidity features  
        if 'q_forecast' in df.columns:
            df['humidity_gradient'] = df['q_forecast'].diff()
            df['relative_humidity'] = df['q_forecast'] / (1 + df['q_forecast']) * 100

        # Atmospheric stability
        if 'w_forecast' in df.columns:
            df['vertical_velocity_abs'] = np.abs(df['w_forecast'])
            df['stability_indicator'] = df['w_forecast'] * df.get('temp_gradient', 0)

        return df

    def _create_lag_features(self, df):
        """Create lagged features for temporal dependencies"""
        lag_cols = ['O3_forecast', 'NO2_forecast', 'T_forecast', 'q_forecast']

        for col in lag_cols:
            if col in df.columns:
                for lag in [1, 2, 3, 6, 12, 24]:
                    df[f'{col}_lag_{lag}h'] = df[col].shift(lag)

        return df

    def _create_rolling_features(self, df):
        """Create rolling window statistics"""
        rolling_cols = ['O3_forecast', 'NO2_forecast', 'T_forecast', 'wind_speed']

        for col in rolling_cols:
            if col in df.columns:
                for window in [3, 6, 12, 24]:
                    df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window).mean()
                    df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window).std()
                    df[f'{col}_rolling_min_{window}h'] = df[col].rolling(window).min()
                    df[f'{col}_rolling_max_{window}h'] = df[col].rolling(window).max()

        return df

    def _create_interaction_features(self, df):
        """Create interaction features between variables"""
        # Temperature-humidity interactions
        if 'T_forecast' in df.columns and 'q_forecast' in df.columns:
            df['temp_humidity_interaction'] = df['T_forecast'] * df['q_forecast']
            df['heat_index'] = df['T_forecast'] + 0.5 * df['q_forecast']

        # Wind-pollution interactions
        if 'wind_speed' in df.columns:
            for col in ['O3_forecast', 'NO2_forecast']:
                if col in df.columns:
                    df[f'{col}_wind_interaction'] = df[col] * df['wind_speed']

        # Satellite-meteorological interactions
        satellite_cols = ['NO2_satellite', 'HCHO_satellite']
        for sat_col in satellite_cols:
            if sat_col in df.columns and 'T_forecast' in df.columns:
                df[f'{sat_col}_temp_interaction'] = df[sat_col] * df['T_forecast']

        return df

    def _get_season(self, month):
        """Get season from month"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn

    def fit_scaler(self, X):
        """Fit scaler on training data"""
        self.scaler.fit(X)
        self.feature_columns = X.columns.tolist()

    def transform(self, X):
        """Transform data using fitted scaler"""
        if self.feature_columns:
            # Ensure same columns as training
            missing_cols = set(self.feature_columns) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            X = X[self.feature_columns]

        return pd.DataFrame(
            self.scaler.transform(X), 
            columns=X.columns, 
            index=X.index
        )

    def fit_transform(self, X):
        """Fit scaler and transform data"""
        self.fit_scaler(X)
        return self.transform(X)

