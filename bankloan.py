import streamlit as st
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# Load the trained model from Hugging Face
model_path = hf_hub_download(repo_id="ifiecas/LoanApproval-DT-v1.0", filename="best_pruned_dt.pkl")
model = joblib.load(model_path)

# Streamlit app title
st.title("🏦 AI-Powered Loan Approval System")
st.write("Enter your details to check your loan approval status.")

# Input fields
applicant_income = st.number_input("Applicant's Monthly Income ($)", min_value=0)
coapplicant_income = st.number_input("Co-Applicant's Monthly Income ($)", min_value=0)
loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0)
loan_term = st.number_input("Loan Term (days)", min_value=0, value=360)
credit_history = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)")
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Married", "Not Married"])
education = st.selectbox("Education Level", ["Graduate", "Under Graduate"])
self_employed = st.selectbox("Self-Employed", ["Yes", "No"]) 
location = st.selectbox("Property Location", ["Urban", "Semiurban", "Rural"])

def preprocess_input():
    # Convert categorical inputs to numerical format (you may need encoding based on your dataset)
    gender_num = 1 if gender == "Male" else 0
    marital_status_num = 1 if marital_status == "Married" else 0
    education_num = 1 if education == "Graduate" else 0
    self_employed_num = 1 if self_employed == "Yes" else 0
    location_num = {"Urban": 2, "Semiurban": 1, "Rural": 0}[location]

    return np.array([[
        applicant_income, coapplicant_income, loan_amount, loan_term, credit_history,
        gender_num, marital_status_num, education_num, self_employed_num, location_num
    ]])

# Predict button
if st.button("Check Loan Approval"):
    input_data = preprocess_input()
    prediction = model.predict(input_data)
    result = "✅ Approved" if prediction[0] == "Y" else "❌ Rejected"
    st.subheader(f"Loan Status: {result}")

st.write("📌 AI-driven decision-making for faster loan approvals.")

# Create requirements.txt for dependencies
requirements = """streamlit
joblib
numpy
scikit-learn
huggingface_hub
"""

with open("requirements.txt", "w") as f:
    f.write(requirements)
