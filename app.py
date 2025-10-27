import streamlit as st
import mlflow.sklearn
import pandas as pd

# ✅ Load Production Model
model_uri = "models:/insurance-cost-predictor/Production"
model = mlflow.sklearn.load_model(model_uri)

# ✅ Streamlit UI
st.title("💰 Medical Insurance Cost Predictor")
st.write("Enter the details below to predict the insurance charges:")

age = st.number_input("Age", min_value=0, max_value=100, value=30)
bmi = st.number_input("BMI Value", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# ✅ Prepare DataFrame input for Model
input_df = pd.DataFrame({
    "sex": [sex],
    "smoker": [smoker],
    "region": [region],
    "age": [age],
    "bmi": [bmi],
    "children": [children]
})

if st.button("Predict Insurance Cost"):
    try:
        prediction = model.predict(input_df)
        st.success(f"✅ Predicted Medical Insurance Cost: ${prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Prediction Failed ❌: {str(e)}")
