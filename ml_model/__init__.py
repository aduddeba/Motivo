import os
import joblib

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "car_recommender.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Model file not found. Run train_model.py first.")
    
    model, scaler, features, encoders, target_encoder = joblib.load(model_path)
    return model, scaler, features, encoders, target_encoder
