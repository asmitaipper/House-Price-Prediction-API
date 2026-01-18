from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import os
import pandas as pd

MODEL_PATH = "artifacts/house_price_model.joblib"

app = FastAPI(title="House Price Prediction API")

class HouseFeatures(BaseModel):
    rooms: int
    area: float
    age: float
    location_score: float

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train the model first."
        )
    return load(MODEL_PATH)

model = load_model()

@app.get("/")
def root():
    return {"message": "House Price Prediction API is running."}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    data = pd.DataFrame([features.dict()])
    prediction = model.predict(data)[0]
    return {"predicted_price": float(prediction)}
