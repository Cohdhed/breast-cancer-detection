import pandas as pd
import numpy as np 
import streamlit as st 
import joblib


artifact = joblib.load('cancer_detection.joblib')

st.title('Breast Cancer Detection App')
st.write("""
This application predicts whether a breast tumor is malignant (cancerous) or benign (non-cancerous).
Please enter the following features of the tumor to get the prediction.
""")

# Define input fields for each feature
radius_mean = st.number_input('Radius Mean', min_value=0.0, max_value=50.0, step=0.1)
texture_mean = st.number_input('Texture Mean', min_value=0.0, max_value=50.0, step=0.1)
smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0, max_value=1.0, step=0.01)
compactness_mean = st.number_input('Compactness Mean', min_value=0.0, max_value=1.0, step=0.01)
concavity_mean = st.number_input('Concavity Mean', min_value=0.0, max_value=1.0, step=0.01)
symmetry_mean = st.number_input('Symmetry Mean', min_value=0.0, max_value=1.0, step=0.01)
fractal_dimension_mean = st.number_input('Fractal Dimension Mean', min_value=0.0, max_value=1.0, step=0.01)
radius_se = st.number_input('Radius Standard Error', min_value=0.0, max_value=5.0, step=0.01)
texture_se = st.number_input('Texture Standard Error', min_value=0.0, max_value=5.0, step=0.01)
smoothness_se = st.number_input('Smoothness Standard Error', min_value=0.0, max_value=0.5, step=0.01)
compactness_se = st.number_input('Compactness Standard Error', min_value=0.0, max_value=0.5, step=0.01)
concavity_se = st.number_input('Concavity Standard Error', min_value=0.0, max_value=0.5, step=0.01)
concave_points_se = st.number_input('Concave Points Standard Error', min_value=0.0, max_value=0.5, step=0.01)
symmetry_se = st.number_input('Symmetry Standard Error', min_value=0.0, max_value=0.5, step=0.01)
fractal_dimension_se = st.number_input('Fractal Dimension Standard Error', min_value=0.0, max_value=0.5, step=0.01)
smoothness_worst = st.number_input('Smoothness Worst', min_value=0.0, max_value=1.0, step=0.01)
compactness_worst = st.number_input('Compactness Worst', min_value=0.0, max_value=1.0, step=0.01)
symmetry_worst = st.number_input('Symmetry Worst', min_value=0.0, max_value=1.0, step=0.01)
fractal_dimension_worst = st.number_input('Fractal Dimension Worst', min_value=0.0, max_value=1.0, step=0.01)



user_input = {
    'radius_mean': radius_mean,
    'texture_mean': texture_mean,
    'smoothness_mean': smoothness_mean,
    'compactness_mean': compactness_mean,
    'concavity_mean': concavity_mean,
    'symmetry_mean': symmetry_mean,
    'fractal_dimension_mean': fractal_dimension_mean,
    'radius_se': radius_se,
    'texture_se': texture_se,
    'smoothness_se': smoothness_se,
    'compactness_se': compactness_se,
    'concavity_se': concavity_se,
    'concave points_se': concave_points_se,
    'symmetry_se': symmetry_se,
    'fractal_dimension_se': fractal_dimension_se,
    'smoothness_worst': smoothness_worst,
    'compactness_worst': compactness_worst,
    'symmetry_worst': symmetry_worst,
    'fractal_dimension_worst': fractal_dimension_worst
}


def check_missing_inputs(data):
    return any(value == '' for value in data.values())


def make_prediction(data):
    df = pd.DataFrame([data])
    X = pd.DataFrame(artifact['preprocessing'].transform(df),
                     columns=artifact['cols'])
    
    prediction = artifact['model'].predict(X)[0]
    decision = 'Malignant (Cancerous)' if prediction == 1 else 'Benign (Non Cancerous)'
    return f"The tumor is {decision}"


# Button to make prediction
if st.button('Predict'):
    if check_missing_inputs(user_input):
        st.error("Please fill in all the fields.")
    else:
        result = make_prediction(user_input)
        st.success(result)