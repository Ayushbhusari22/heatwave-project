# prediction/predict.py
import pickle
import pandas as pd
from geopy.geocoders import Nominatim
import requests

class HeatwavePredictor:
    def __init__(self, model_path, scaler_path):
        """Lightweight prediction-only version"""
        self.model = pickle.load(open(model_path, 'rb'))
        self.scaler = pickle.load(open(scaler_path, 'rb'))
        self.features = [
            'temperature_2m_max', 'apparent_temperature_max',
            'relative_humidity_2m_mean', 'wind_speed_10m_max',
            'pressure_msl_mean', 'precipitation_sum', 'cloud_cover_mean'
        ]
        self.geolocator = Nominatim(user_agent="heatwave_predictor")

    def predict(self, city, days=7):
        """Main prediction interface"""
        forecast = self._get_forecast(city, days)
        processed = self._process_features(forecast)
        return self._generate_predictions(processed)

    def _get_forecast(self, city, days):
        """Fetch forecast data from API"""
        try:
            lat, lon = self._get_coordinates(city)
            response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": self.features,
                    "forecast_days": days,
                    "timezone": "auto"
                },
                timeout=10
            )
            return pd.DataFrame(response.json()["daily"])
        except Exception as e:
            raise ValueError(f"Forecast fetch failed: {str(e)}")

    def _process_features(self, df):
        """Replicate training preprocessing"""
        X = df[self.features].copy()
        X['temp_humidity_ratio'] = X['temperature_2m_max'] / X['relative_humidity_2m_mean'].clip(lower=1)
        X['diurnal_range'] = X['temperature_2m_max'] - X['apparent_temperature_max']
        return self.scaler.transform(X)

    def _generate_predictions(self, features):
        """Generate probabilities and classifications"""
        proba = self.model.predict_proba(features)[:, 1]
        return {
            'probability': proba,
            'is_heatwave': (proba > 0.5).astype(int)
        }