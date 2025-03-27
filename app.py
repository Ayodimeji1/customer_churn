
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- Load model and scaler from provided pickle files ---
with open('customer_churn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# --- Set page config for a clean layout and title ---
st.set_page_config(page_title="Customer Churn Predictor", 
                   page_icon="🔍", 
                   layout="centered")

# --- Stylish Title & Description ---
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🔍 Customer Churn Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #7F8C8D;'>Enter customer details below to predict if the customer is likely to churn.</h4>", unsafe_allow_html=True)
st.markdown("---")

# --- Three-column layout with wider columns and stylish subheaders ---
col1, empty_col, col2, empty_col2, col3 = st.columns([1.2, 0.2, 1.2, 0.2, 1.2])

with col1:
    st.markdown("### 📄 Personal Info")
    credit_score = st.slider("Credit Score", min_value=300, max_value=950, value=600) 
    age = st.slider("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=5)   
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"], index=0)
    
with col2:
    st.markdown("### 💳 Financial Info")
    balance = st.slider("Balance (£)", min_value=0.0, max_value=1000000.0, value=60000.0, format="%0.0f")
    estimated_salary = st.slider("Estimated Salary (£)", min_value=0.0, max_value=500000.0, value=50000.0, format="%0.0f")
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4, 5])
    has_cr_card = st.radio("Has Credit Card", ["No", "Yes"], index=1)
    card_type = st.selectbox("Card Type", ["Blue", "Silver", "Gold", "Platinum"])
    point_earned = st.slider("Point Earned", min_value=0, max_value=1000, value=200, format="%0.0f")
   
with col3:
    st.markdown("### 🔐 Account & Churn Details")
    is_active_member = st.radio("Is Active Member", ["No", "Yes"], index=1)
    complain = st.radio("Complain", ["No", "Yes"], index=0)
    satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)

# --- Encode Categorical Features ---
geography_germany = int(geography == "Germany")
geography_spain = int(geography == "Spain")
gender_male = int(gender == "Male")
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0
complain = 1 if complain == "Yes" else 0
card_type_mapping = {"Blue": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
card_type_encoded = card_type_mapping[card_type]

# Prepare input data
input_data = pd.DataFrame([[
    credit_score, age, tenure, balance, num_of_products, has_cr_card, is_active_member,
    estimated_salary, complain, satisfaction_score, card_type_encoded, point_earned,
    geography_germany, geography_spain, gender_male
]], columns=[
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Complain", "Satisfaction Score", "Card Type", "Point Earned",
    "Geography_Germany", "Geography_Spain", "Gender_Male"
])

# Scale the input data
input_scaled = scaler.transform(input_data)

# --- Display Prediction in the Last Column ---
with col3:
    st.markdown("<h3 style='color: #2ECC71;'>Prediction Section</h3>", unsafe_allow_html=True)
    if st.button("🔮 Predict"):
        churn_prob = model.predict_proba(input_scaled)[0][1]  # probability of churn
        churn_pred = model.predict(input_scaled)[0]           # predicted class (0 or 1)

        st.markdown("---")
        if churn_pred == 1:
            st.error(f"❌ Prediction: Customer will likely churn with {churn_prob * 100:.2f}% confidence.")
        else:
            st.success(f"✅ Prediction: Customer will likely stay with {(1 - churn_prob) * 100:.2f}% confidence.")
