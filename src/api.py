from fastapi import FastAPI
import pandas as pd
import mlflow
import os

app = FastAPI()

# CRITICAL: connect to MLflow server running on host
mlflow.set_tracking_uri("http://host.docker.internal:5000")

MODEL_NAME = "customer-churn"

# Load production model from MLflow Registry
model = mlflow.pyfunc.load_model("models:/customer-churn@production")

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)

    return {
        "prediction": int(prediction[0]),
        "churn": "Yes" if prediction[0] == 1 else "No"
    }
