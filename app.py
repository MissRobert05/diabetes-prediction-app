import streamlit as st
import pandas as pd
import joblib
import base64
import os

# -------------------------------
# 🎨 Background Image Function
# -------------------------------
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded_string = base64.b64encode(img.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)),
                    url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
        color: white;
    }}
    .title-text {{
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
    }}
    .footer {{
        font-size: 0.8rem;
        text-align: center;
        padding: 1rem;
        color: #ccc;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -------------------------------
# 🧠 Load Models
# -------------------------------
@st.cache_resource
def load_models():
    model_dir = "models"
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    models = {}
    for file in model_files:
        name = file.replace("_", " ").replace(".pkl", "").title()
        models[name] = joblib.load(os.path.join(model_dir, file))
    return models

# -------------------------------
# 📋 Input Form Function
# -------------------------------
def user_input_form():
    st.subheader("📝 Enter Patient Details:")
    gender = st.selectbox("Gender", ['Female', 'Male', 'Other'])
    age = st.number_input("Age", 1, 120, 30)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    smoking_history = st.selectbox("Smoking History", ['never', 'former', 'current', 'not current', 'ever', 'No Info'])
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 6.0)
    glucose = st.number_input("Blood Glucose Level", 50, 300, 120)

    df = pd.DataFrame([[gender, age, hypertension, heart_disease,
                        smoking_history, bmi, hba1c, glucose]],
                      columns=['gender', 'age', 'hypertension', 'heart_disease',
                               'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])
    return df

# -------------------------------
# 🚀 Main App
# -------------------------------
def main():
    st.set_page_config(page_title="Diabetes Predictor", layout="centered")
    set_background("diabetes-bg.jpg")

    st.markdown('<div class="title-text">🧬 Diabetes Prediction App</div>', unsafe_allow_html=True)
    st.markdown("Welcome! This app uses multiple machine learning models to predict the likelihood of diabetes.")

    models = load_models()
    encoder = joblib.load("ordinal_encoder.pkl")

    df = user_input_form()

    # Ordinal Encode 'gender' and 'smoking_history'
    df[['gender', 'smoking_history']] = encoder.transform(df[['gender', 'smoking_history']])

    selected_model_name = st.selectbox("🤖 Select a Model", list(models.keys()))
    model = models[selected_model_name]

    if st.button("🔍 Predict"):
        prediction = model.predict(df)[0]
        result = "⚠️ Likely Diabetic" if prediction == 1 else "✅ Not Diabetic"
        st.success(f"**Result using {selected_model_name}:** {result}")

    st.markdown('<div class="footer">Built with ❤️ by MISS ROBERT</div>', unsafe_allow_html=True)

# -------------------------------
# ▶️ Run the App
# -------------------------------
if __name__ == "__main__":
    main()
