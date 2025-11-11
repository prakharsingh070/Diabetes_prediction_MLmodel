# app.py
import streamlit as st
import pickle
import numpy as np

# Load the saved model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🩺 Diabetes Prediction App")
st.write("Enter the patient's details below to predict diabetes risk:")

# User inputs
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
Glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
Age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Convert inputs into numpy array
features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigreeFunction, Age]])

# Scale the features
scaled_features = scaler.transform(features)

# Prediction
if st.button("Predict"):
    prediction = model.predict(scaled_features)
    if prediction[0] == 1:
        st.error("🚨 The patient is likely to have diabetes.")
    else:
        st.success("✅ The patient is not likely to have diabetes.")
