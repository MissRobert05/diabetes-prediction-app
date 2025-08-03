# 🩺 Diabetes Prediction App

This project is a **Machine Learning web app** that predicts whether a person is likely to have **diabetes** based on clinical features. The app is built with **Python**, trained in **Google Colab** using **scikit-learn**, and deployed using **Streamlit**.

---


## 🚀 Features
- Trained multiple models: **Logistic Regression**, **Random Forest**, **Decision Tree**, **Support Vector Machine**, **K-Nearest Neighbors**, and **Naive Bayes**.
- Encoded categorical features using **Ordinal Encoding**.
- Visualized data distributions, ROC Curves, Confusion Matrices, and Feature Importance during training.
- Saved trained models using `joblib` with compression.
- User-friendly **Streamlit interface** for live predictions.
- Beautiful background with a responsive form for user inputs.
- Selectable models for prediction and dynamic result display.

---


## 📓 Model Training
The machine learning models were trained and evaluated in **Google Colab**. After training, the models and preprocessing encoders (OrdinalEncoder & Scaler) were saved as `.pkl` files and stored in the `/models` directory for deployment.  
> **Note:** The training notebook is not included in this repository as only the deployment artifacts are required.

---


## 📂 Project Structure
Diabetes-Prediction-App/
│
├── app.py
├── models/
│ ├── logistic_regression_model.pkl
│ ├── random_forest_model.pkl
│ ├── decision_tree_model.pkl
│ ├── svm_model.pkl
│ ├── knn_model.pkl
│ ├── naive_bayes_model.pkl
│ ├── scaler.pkl
│
├── gender_encoder.pkl
├── diabetes-bg.jpg
├── requirements.txt
├── README.md
└── .gitignore


---


## 🛠️ How to Run Locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com/MissRobert05/diabetes-prediction-app.git
   cd diabetes-prediction-app

2. **Create a virtual environment:**
bash:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies:**
bash:
pip install -r requirements.txt

4. **Run the Streamlit App:**
bash:
streamlit run app.py



🌐 Deployment (Optional)
You can deploy this app on:

Streamlit Cloud

Render.com

Heroku

Vercel (Streamlit deployment guide)

For Streamlit Cloud Deployment:

Push this repository to GitHub.

Go to Streamlit Cloud.

Connect your GitHub repo.

Select app.py as the entry file.

Click Deploy.



✍️ Author
MISS ROBERT



📝 License
This project is open-source and free to use.

.gitignore
Virtual environment
venv/
.env

Python cache
pycache/
*.pyc

System files
.DS_Store
Thumbs.db

Streamlit cache
.streamlit/

Models and large files (if you don't want to track them)
models/
Jupyter Notebook checkpoints
.ipynb_checkpoints/

PyCharm & VSCode
.idea/
.vscode/

Log files
*.log

Mac system files
.DS_Store

---