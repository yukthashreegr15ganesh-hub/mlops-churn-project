import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load pipeline
pipeline = joblib.load(PROJECT_ROOT / "model" / "pipeline.pkl")

# Load new data
df = pd.read_csv(PROJECT_ROOT / "data" / "churn.csv")

X = df.drop(["Churn", "customerID"], axis=1)

# Ensure TotalCharges is numeric and drop NA to match training preprocessing
X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
X = X.dropna()

# Predict using the full pipeline
preds = pipeline.predict(X)
print("Predictions:", preds[:10])
