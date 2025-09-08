from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Load your model
model = joblib.load("model.pkl")

# Define input structure
class VehicleRequest(BaseModel):
    make_model: str

# FastAPI app setup
app = FastAPI()

# Allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict_rating(data: VehicleRequest):
    # Replace with real model input processing
    input_text = data.make_model.lower()
    
    # Dummy rating logic (replace with real logic)
    rating = 4.5 if "civic" in input_text else 3.0
    recommendations = ["Toyota Corolla", "Mazda 3", "Hyundai Elantra"] if "civic" in input_text else []

    return {
        "rating": rating,
        "recommendations": recommendations
    }
