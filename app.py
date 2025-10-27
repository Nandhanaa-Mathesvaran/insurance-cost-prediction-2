import streamlit as st
import numpy as np
import pickle
import pandas as pd

st.title("Medical Insurance Cost Predictor")

models = {
    "GradientBoosting": "insurance_GradientBoosting.pkl",
    "RandomForest": "insurance_RandomForest.pkl",
    "XGBRegressor": "insurance_XGBRegressor.pkl"
}

model_choice = st.selectbox("Select a Model", list(models.keys()))

pkl_file = models[model_choice]
model = pickle.load(open(pkl_file, "rb"))

age = st.number_input("Age", min_value=0, max_value=120, value=30)
bmi = st.number_input("BMI Value", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Predict Insurance Cost"):
    input_data = pd.DataFrame([[sex, smoker, region, age, bmi, children]],
                              columns=['sex', 'smoker', 'region', 'age', 'bmi', 'children'])

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Medical Insurance Cost: ${round(prediction, 2)}")
