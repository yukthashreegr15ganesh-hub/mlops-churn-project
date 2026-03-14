from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

pipeline = joblib.load("model/pipeline.pkl")


# Define input schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}


@app.post("/predict")
def predict(data: CustomerData):

    # Convert request to dataframe
    df = pd.DataFrame([data.model_dump()])

    # Force correct numeric types
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["tenure"] = df["tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    prediction = pipeline.predict(df)[0]

    return {"prediction": prediction}