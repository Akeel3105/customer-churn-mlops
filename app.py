import mlflow
import mlflow.sklearn
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Create FastAPI application
app = FastAPI()


# Load trained model from MLflow Registry
#model = mlflow.sklearn.load_model(
#    "model"
# 
#)

model = joblib.load("model.pkl")


# Input structure
class PredictionRequest(BaseModel):
    features: list[float]


# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):

    # Check number of features
    if len(request.features) != 30:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 30 features, but received {len(request.features)}"
        )

    # Make prediction
    prediction = model.predict([request.features])

    return {
        "prediction": int(prediction[0])
    }