import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("data/churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

X = df.drop(["Churn", "customerID"], axis=1)
y = df["Churn"]

# Identify column types
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Model
model = RandomForestClassifier(n_estimators=100)

# Full pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)

# MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("customer-churn-exp")

with mlflow.start_run():

    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name="customer-churn"
    )

print("Model accuracy:", accuracy)

# Save pipeline locally for API
os.makedirs("model", exist_ok=True)

joblib.dump(pipeline, "model/pipeline.pkl")

print("Pipeline saved to model/pipeline.pkl")