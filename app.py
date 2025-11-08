import streamlit as st
import joblib
import numpy as np

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="centered"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        text-align: center;
        color: #FF4B4B;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        color: black;
    }
    .stMarkdown {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title and Banner ---
st.title("❤️ AI Heart Disease Prediction App")
st.image("heart.png", width="stretch")
st.markdown(
    "<h4 style='text-align:center;'>Predict your heart disease risk with Machine Learning</h4>",
    unsafe_allow_html=True
)
st.write(
    "This web app uses AI and medical data to predict your likelihood of heart disease. "
    "*(For educational and demonstration purposes only)*"
)

# --- Load Model & Scaler ---
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Input Section ---
st.header("🩺 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", ("Male", "Female"))
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=1)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("No", "Yes"))

with col2:
    restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, value=1)
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", ("No", "Yes"))
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.number_input("Slope of Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=1)
    ca = st.number_input("Major Vessels Colored (0-4)", min_value=0, max_value=4, value=0)
    thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3, value=2)

# --- Encode categorical values ---
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# --- Predict Button ---
if st.button("🔍 Predict Heart Disease Risk"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Result:")
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease! (Probability: {probability*100:.2f}%)")
        st.markdown(
            "<p style='color:#FF4B4B; font-size:18px;'>Please consult a cardiologist for further medical evaluation.</p>",
            unsafe_allow_html=True
        )
    else:
        st.success(f"✅ Low Risk of Heart Disease (Probability: {probability*100:.2f}%)")
        st.markdown(
            "<p style='color:#00FFAA; font-size:18px;'>Great! Maintain a healthy lifestyle and regular checkups 💪</p>",
            unsafe_allow_html=True
        )

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>© 2025 AI Health Prediction | Built with ❤️ by Kumar GK</p>",
    unsafe_allow_html=True
)
