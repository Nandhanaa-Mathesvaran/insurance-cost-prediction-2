import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import mlflow
import mlflow.sklearn

# Load dataset
df = pd.read_csv("insurance.csv")

# Features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Identify categorical columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Preprocessing: OneHotEncoding for categorical columns
preprocess = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop="first"), cat_cols)],
    remainder="passthrough"
)

# Model pipeline: Gradient Boosting instead of Random Forest
model = Pipeline([
    ("preprocess", preprocess),
    ("regressor", GradientBoostingRegressor(random_state=42))
])

# Start MLflow experiment
mlflow.set_experiment("Insurance-Model-Tracking")

with mlflow.start_run():
    model.fit(X, y)

    preds = model.predict(X)

    r2 = r2_score(y, preds)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)

    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(model, "insurance_model")

print("Training Completed & Logged to MLflow!")
print(f"R2 Score: {r2}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
