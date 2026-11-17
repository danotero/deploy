import streamlit as st
import requests
import os

# The API URL will be passed as an environment variable in Azure
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.title("Iris Species Predictor")
st.write("Enter the flower dimensions to predict its species.")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 5.0, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 3.0, 1.0)

if st.button("Predict"):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        prediction = response.json().get("prediction")
        st.success(f"The predicted species is: **{prediction}**")
    except Exception as e:
        st.error(f"Error connecting to the API: {e}")
