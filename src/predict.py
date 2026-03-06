import pandas as pd
import joblib

# Load model & features
model = joblib.load("model/model.pkl")
features = joblib.load("model/features.pkl")

# Load new data
df = pd.read_csv("data/churn.csv")

X = df.drop("Churn", axis=1)

X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
X = X.fillna(0)

# One-hot encode
X = pd.get_dummies(X)

# Align columns
X = X.reindex(columns=features, fill_value=0)

# Predict
preds = model.predict(X)
print(preds[:10])
