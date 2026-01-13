import streamlit as st
import pandas as pd
import joblib

# Load trained Random Forest model
model = joblib.load("churn_model.pkl")

# App title
st.title("Customer Churn Prediction")

st.markdown("""
Enter customer details below to predict the probability of churn.
""")

# User inputs
age = st.slider("Age", min_value=18, max_value=80, value=35)
tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
monthly = st.number_input("Monthly Charges", min_value=20.0, max_value=120.0, value=50.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber"])
payment = st.selectbox("Payment Method", ["Card", "Bank Transfer", "Cash"])

# Automatically calculate Total Charges
total_charges = round(monthly * tenure, 2)

# Prepare input DataFrame matching pipeline features
input_df = pd.DataFrame([{
    "Age": age,
    "Tenure_Months": tenure,
    "Monthly_Charges": monthly,
    "Total_Charges": total_charges,
    "Contract_Type": contract,
    "Internet_Service": internet,
    "Payment_Method": payment
}])

# Prediction button
if st.button("Predict Churn Probability"):
    prob = model.predict_proba(input_df)[0][1]
    prediction = "High Risk of Churn" if prob >= 0.5 else "Low Risk of Churn"
    
    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** {prob:.2%}")
    st.write(f"**Risk Level:** {prediction}")

    if prob >= 0.5:
        # High risk displayed in red
        st.error("This customer is likely to churn. Consider retention actions.")
    else:
        # Low risk displayed in green
        st.success("This customer is unlikely to churn. Keep engagement consistent.")
