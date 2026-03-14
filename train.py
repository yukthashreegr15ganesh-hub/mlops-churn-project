import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/churn.csv")

X = df.drop("churn", axis=1)
y = df["churn"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model/model.pkl")

print("Model saved to model/model.pkl")