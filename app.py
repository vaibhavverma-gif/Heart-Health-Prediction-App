import streamlit as st
import pandas as pd
import joblib

model=joblib.load('LR_Heart_Disease_predictor.pkl')
scaler=joblib.load('scaler.pkl')
expected_columns=joblib.load('columns.pkl')


st.title("Heart Disease Prediction App by Vaibhav❤️")
st.markdown("Provide the required details to predict heart disease risk.")

age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.slider("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 1000, 200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl",[0,1])
Resting_ECG = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST", "LVH"])
max_heart_rate = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
st_slope = st.selectbox("Slope of ST Segment", ["Up", "Flat", "Down"])


if st.button("Predict"):
    raw_data = {
        'Age': age,
        'Sex' + sex : 1,
        'ChestPainType' + chest_pain : 1,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_blood_sugar,
        'RestingECG' + Resting_ECG : 1,
        'MaxHR': max_heart_rate,
        'ExerciseAngina' + exercise_angina : 1,
        'Oldpeak': oldpeak,
        'ST_Slope' + st_slope : 1,
    }

    input_data = pd.DataFrame([raw_data])

    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0


    input_df = input_data[expected_columns]

    scaled_input = scaler.transform(input_df)  
    prediction = model.predict(scaled_input)[0] 

    if prediction == 1:
        st.error("⚠️You are at risk of heart disease. Please consult a healthcare professional.")
    else:
        st.success("✅You are not at risk of heart disease. Keep up the healthy lifestyle!")