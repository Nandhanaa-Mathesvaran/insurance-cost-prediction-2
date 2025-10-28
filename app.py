import streamlit as st
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np

# MLflow model details (make sure you logged your Gradient Boosting model with this name)
MODEL_NAME = "insurance-cost-predictor-gradientboosting"
MODEL_STAGE = "Production"

# Load the model from MLflow Model Registry
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# Streamlit UI
st.title(" Medical Insurance Cost Predictor (Gradient Boosting Model)")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI Value", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Predict Insurance Cost"):
    try:
        # Create a DataFrame from user input
        input_data = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "children": children,
            "sex": sex,
            "smoker": smoker,
            "region": region
        }])

        # Make prediction
        prediction = model.predict(input_data)[0]

        st.success(f" Predicted Insurance Cost: ${prediction:.2f}")

    except Exception as e:
        st.error(f" Prediction Failed: {str(e)}")
