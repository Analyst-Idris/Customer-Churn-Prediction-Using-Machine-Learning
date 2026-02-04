import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("churn_model.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("Customer Churn Prediction App")
st.write("Predict the likelihood of customer churn using a machine learning model.")

st.divider()

# -------------------- User Inputs --------------------
age = st.slider("Age", 18, 80, 35)
tenure = st.slider("Tenure (Months)", 0, 72, 12)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0,
    help="Total amount billed to the customer so far"
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber"]
)

payment = st.selectbox(
    "Payment Method",
    ["Card", "Bank Transfer", "Cash"]
)

# -------------------- Input DataFrame --------------------
input_df = pd.DataFrame([{
    "Age": age,
    "Tenure_Months": tenure,
    "Monthly_Charges": monthly,
    "Total_Charges": total_charges,
    "Contract_Type": contract,
    "Internet_Service": internet,
    "Payment_Method": payment
}])

# -------------------- Prediction --------------------
if st.button("Predict Churn Risk"):
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** {prob:.2%}")

    if prob >= 0.7:
        st.error("🔴 High Risk of Churn\n\nImmediate retention action required.")
    elif prob >= 0.4:
        st.warning("🟠 Medium Risk of Churn\n\nMonitor and engage proactively.")
    else:
        st.success("🟢 Low Risk of Churn\n\nCustomer is unlikely to churn.")

