import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set MLflow tracking URI (local server)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("customer-churn-exp")

# Load dataset
df = pd.read_csv("data/Churn.csv")

# ================= CLEANING (VERY IMPORTANT) =================

# 1. Drop customerID (string ID column)
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# 2. Fix TotalCharges (has blank strings)
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 3. Drop missing values
df = df.dropna()

# 4. Convert target column (Churn) to 0/1
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 5. One-hot encode categorical features (THIS FIXES 'Female' ERROR)
df = pd.get_dummies(df, drop_first=True)

# ================= FEATURES / TARGET =================
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= MLflow Training =================
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log metrics
    mlflow.log_metric("accuracy", acc)

    # Log model to MLflow Registry
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="customer-churn"
    )

    print(f"Model Accuracy: {acc}")
import joblib
import os

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/model.pkl")

print("Model saved to model/model.pkl")
