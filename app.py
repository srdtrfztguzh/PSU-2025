import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ✅ Load the transformation pipeline
try:
    with open('models/data_transformation.pkl', 'rb') as file:
        transformer = pickle.load(file)
except FileNotFoundError:
    st.error("Data transformation file not found!")
    st.stop()

# ✅ Load the ensemble model
model_path = 'models/ensemble_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
else:
    st.error("Ensemble model file not found!")
    st.stop()

# ✅ Title
st.title('Chronic Kidney Disease Prediction (Ensemble Learning)')

# ✅ Create input fields for user input
age = st.number_input('Age', min_value=0, max_value=120, value=50)
bp = st.number_input('Blood Pressure (in mmHg)', min_value=0, max_value=200, value=80)
sg = st.number_input('Specific Gravity', min_value=1.000, max_value=1.030, value=1.015)
al = st.number_input('Albumin', min_value=0, max_value=5, value=0)
su = st.number_input('Sugar', min_value=0, max_value=5, value=0)
rbc = st.selectbox('Red Blood Cells', ['normal', 'abnormal'])
pc = st.selectbox('Pus Cell', ['normal', 'abnormal'])
pcc = st.selectbox('Pus Cell Clumps', ['present', 'notpresent'])
ba = st.selectbox('Bacteria', ['present', 'notpresent'])
bgr = st.number_input('Blood Glucose Random (in mg/dl)', min_value=0, max_value=500, value=100)
bu = st.number_input('Blood Urea (in mg/dl)', min_value=0, max_value=300, value=30)
sc = st.number_input('Serum Creatinine (in mg/dl)', min_value=0.0, max_value=30.0, value=1.2)
sod = st.number_input('Sodium (in mEq/L)', min_value=0, max_value=200, value=135)
pot = st.number_input('Potassium (in mEq/L)', min_value=0.0, max_value=15.0, value=4.5)
hemo = st.number_input('Hemoglobin (in g/dl)', min_value=0.0, max_value=20.0, value=14.0)
pcv = st.number_input('Packed Cell Volume', min_value=0, max_value=100, value=45)
wc = st.number_input('White Blood Cell Count (in cells/cumm)', min_value=0, max_value=50000, value=7500)
rc = st.number_input('Red Blood Cell Count (in millions/cmm)', min_value=0.0, max_value=10.0, value=4.5)
htn = st.selectbox('Hypertension', ['yes', 'no'])
dm = st.selectbox('Diabetes Mellitus', ['yes', 'no'])
cad = st.selectbox('Coronary Artery Disease', ['yes', 'no'])
appet = st.selectbox('Appetite', ['good', 'poor'])
pe = st.selectbox('Pedal Edema', ['yes', 'no'])
ane = st.selectbox('Anemia', ['yes', 'no'])

# ✅ Create dataframe with original string values
data = {
    'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
    'rbc': rbc, 'pc': pc, 'pcc': pcc, 'ba': ba, 
    'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot,
    'hemo': hemo, 'pcv': pcv, 'wc': wc, 'rc': rc,
    'htn': htn, 'dm': dm, 'cad': cad, 'appet': appet, 
    'pe': pe, 'ane': ane
}

# ✅ Create dataframe
input_df = pd.DataFrame([data])

# ✅ Ensure column values match transformer expectations
for col in input_df.columns:
    if col in transformer.named_transformers_:
        expected_categories = transformer.named_transformers_[col].categories_[0]
        input_df[col] = input_df[col].apply(lambda x: x if x in expected_categories else expected_categories[0] if expected_categories else 'Unknown')

# ✅ Transform the input
try:
    input_scaled = transformer.transform(input_df)
    st.success("Input data transformed successfully!")
except ValueError as e:
    st.error(f"Transformation Error: {e}")
    st.stop()

# ✅ Predict button
if st.button('Predict'):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error('⚠️ Positive: Chronic Kidney Disease Detected!')
    else:
        st.success('✅ Negative: No Chronic Kidney Disease Detected')
