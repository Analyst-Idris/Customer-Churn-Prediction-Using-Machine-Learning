import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Import all sklearn classes used in the training pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# ----------------------------
# Load the trained model safely
# ----------------------------
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "churn_model.pkl")
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Make sure 'churn_model.pkl' is in the app folder.")
        st.stop()
    except AttributeError as e:
        st.error(f"Error loading model: {e}. Make sure all sklearn classes used are imported.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()

model = load_model()

# ----------------------------
# Streamlit App UI
# ----------------------------
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
    try:
        prob = model.predict_proba(input_df)[0][1]
        prediction = "High Risk of Churn" if prob >= 0.5 else "Low Risk of Churn"

        st.subheader("Prediction Result")
        st.write(f"**Churn Probability:** {prob:.2%}")
        st.write(f"**Risk Level:** {prediction}")

        if prob >= 0.5:
            st.error("This customer is likely to churn. Consider retention actions.")
        else:
            st.success("This customer is unlikely to churn. Keep engagement consistent.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
