# House Price Prediction API (End-to-End ML)

End-to-end House Price Prediction system that trains a regression model on housing data and serves predictions via a FastAPI API.  
Includes model training script, saved model artifacts, Docker containerization, and a simple CI workflow.

## Features
- Data loading and preprocessing from CSV housing dataset.
- Training of a regression model (e.g., RandomForestRegressor) with basic evaluation (MAE, RMSE, R2).
- Saving the trained model to disk using joblib.
- FastAPI app that exposes a `/predict` endpoint for house price prediction.
- Dockerfile to containerize the API service.

## How to run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
