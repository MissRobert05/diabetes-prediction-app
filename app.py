import streamlit as st
import pandas as pd
import joblib
import base64
import os
import io

# -------------------------------
# 🎨 Background Image Function
# -------------------------------
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded_string = base64.b64encode(img.read()).decode()
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');

    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)),
                    url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
        color: #FF4B4B;
        font-family: 'Poppins', sans-serif;
    }}
    .title-text {{
        font-size: 3rem;
        font-weight: 600;
        text-align: center;
        padding: 1rem;
        color: #FF4B4B;
    }}
    .footer {{
        font-size: 0.8rem;
        text-align: center;
        padding: 1rem;
        color: #FF4B4B;
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
    st.markdown("This app uses Machine Learning to predict the likelihood of diabetes.")

    models = load_models()
    encoder = joblib.load("ordinal_encoder.pkl")

    df = user_input_form()

    # Ordinal Encode 'gender' and 'smoking_history'
    df[['gender', 'smoking_history']] = encoder.transform(df[['gender', 'smoking_history']])

    selected_model_name = st.selectbox("🤖 Select a Model", list(models.keys()))
    model = models[selected_model_name]

    if st.button("🔍 Predict"):
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        result = "⚠️ Likely Diabetic" if prediction == 1 else "✅ Not Diabetic"
        st.success(f"**Result using {selected_model_name}:** {result}")

    # Future: Database Storage Section (Placeholder)
    st.markdown("---")
    st.info("🔐 *Coming Soon*: Secure Database Connection to store and track user predictions.")

    st.markdown('<div class="footer">Built with ❤️ by MISS ROBERT</div>', unsafe_allow_html=True)

# -------------------------------
# ▶️ Run the App
# -------------------------------
if __name__ == "__main__":
    main()
