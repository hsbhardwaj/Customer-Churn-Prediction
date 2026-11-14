
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom light theme styling
st.markdown("""
<style>
    .main {
        background-color: #F7F9FC;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    .dashboard-title {
        font-size: 38px;
        font-weight: 750;
        color: #172554;
        margin-bottom: 5px;
    }

    .dashboard-subtitle {
        font-size: 16px;
        color: #64748B;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #1E293B;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    div[data-testid="stForm"] {
        background-color: white;
        padding: 25px;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
    }

    div.stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 9px;
        border: none;
        padding: 10px 25px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard header
st.markdown("""
<div class="dashboard-title">
    📊 Customer Churn Prediction
</div>
<div class="dashboard-subtitle">
    AI-powered telecom customer retention analytics
    | Machine Learning Project
</div>
""", unsafe_allow_html=True)

st.divider()

# Model Performance Section
st.markdown('<div class="section-title">📈 Model Performance</div>',
            unsafe_allow_html=True)

st.caption("Evaluation results on the untouched test dataset")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "75.16%")
col2.metric("Precision", "52.26%")
col3.metric("Recall", "74.06%")
col4.metric("ROC-AUC", "82.62%")

st.divider()


# Model Information
with st.expander("ℹ️ About This Machine Learning Model"):
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Problem Type:** Binary Classification")
    st.write("**Dataset:** Telco Customer Churn")
    st.write("**Decision Threshold:** 0.30 (30%)")
    st.write("**Evaluation:** Stratified train-validation-test split")
    st.write("**Preprocessing:** StandardScaler + OneHotEncoder")


# Load trained model and information
model = joblib.load("customer_churn_model.pkl")
model_info = joblib.load("model_info.pkl")


st.subheader("Enter Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )
    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )
    device = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )
    tech = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )
    tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )
    movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    paperless = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

col3, col4 = st.columns(2)

with col3:
    monthly = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

with col4:
    total = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=20000.0,
        value=840.0
    )

st.divider()

if st.button("Predict Customer Churn", type="primary"):

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    probability = model.predict_proba(customer)[0][1]

    threshold = model_info["threshold"]

    prediction = int(probability >= threshold)

    
    st.subheader("📊 Prediction Result")

    churn_percentage = probability * 100

    st.metric(
        "Churn Probability",
        f"{churn_percentage:.2f}%"
    )

    # Risk classification
    if churn_percentage < 15:
        risk_level = "Low Risk"
        st.success("🟢 Low Churn Risk")

    elif churn_percentage < 30:
        risk_level = "Medium Risk"
        st.warning("🟡 Medium Churn Risk")

    else:
        risk_level = "High Risk"
        st.error("🔴 High Churn Risk")

    st.progress(float(probability))

    # Actual model prediction
    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
        st.write("Consider investigating retention opportunities.")
    else:
        st.success("✅ Customer is likely to stay")
        st.write("Customer has a lower predicted churn risk.")

    st.caption(
        f"Risk Category: {risk_level} | "
        f"Model Decision Threshold: {threshold * 100:.0f}%"
    )

# Project Footer
st.divider()

st.markdown("""
<div style="text-align:center; color:#64748B; padding:15px;">
    <b>Telecom Customer Churn Prediction</b><br>
    Built with Python, Scikit-learn and Streamlit<br>
    Machine Learning Portfolio Project
</div>
""", unsafe_allow_html=True)


