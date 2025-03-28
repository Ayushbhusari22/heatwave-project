# prediction/model_loader.py
import os
from .predict import HeatwavePredictor

def load_predictor():
    """Load model from package-relative paths"""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return HeatwavePredictor(
        os.path.join(dir_path, "heatwave_model.pkl"),
        os.path.join(dir_path, "heatwave_scaler.pkl")
    )