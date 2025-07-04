import streamlit as st
import pandas as pd
#import pickle

# -------------------------------
# Load the trained pipeline
# -------------------------------
#model = pickle.load(open('model.pkl','rb'))  # This includes preprocessing + StackingClassifier
import joblib

model = joblib.load("heart.pkl")  # Not pickle.load()
# -------------------------------
# Set up the app interface
# -------------------------------
st.set_page_config(page_title="Smart Heart Risk Classifier", layout="centered")
st.title("💓 Smart Heart Risk Classifier")
st.write("Predict your risk of heart disease based on key health indicators.")

# -------------------------------
# Define the input form
# -------------------------------
with st.form("input_form"):
    age = st.slider("Age", 18, 100)
    sex = st.selectbox("Sex", ["male", "female"])
    cp = st.selectbox("Chest Pain Type", ["typical", "atypical", "non-anginal", "asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 400)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["true", "false"])
    restecg = st.selectbox("Resting ECG Result", ["normal", "ST-T abnormality", "LV hypertrophy"])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 250)
    exang = st.selectbox("Exercise Induced Angina", ["yes", "no"])
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST", ["upsloping", "flat", "downsloping"])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"])

    submit = st.form_submit_button("Predict")
    age_chol=(age * chol)
    thalch_oldpeak= ( thalach/(oldpeak + 1e-5))
#df['age_chol'] = df['age'] * df['chol']
#df['thalch_oldpeak'] = df['thalch'] / (df['oldpeak'] + 1e-5)
# -------------------------------
# Run Prediction on User Input
# -------------------------------
if submit:
    # Package inputs into a single-row DataFrame
    user_input = pd.DataFrame([{
        'age_chol': age_chol,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'fbs': fbs,
        'restecg': restecg,
        'thalch_oldpeak': thalch_oldpeak,
        'exang': exang,

        'slope': slope,
        'ca': ca,
        'thal': thal
    }])

    # Use the model to predict
    probability = model.predict_proba(user_input)[0][1]
    prediction = model.predict(user_input)[0]

    # Display result
    st.markdown(f"### 🧠 Predicted Risk Score: **{probability:.2f}**")
    if prediction == 1:
        st.error("🔴 High likelihood of heart disease detected.")
    else:
        st.success("🟢 Low likelihood of heart disease.")