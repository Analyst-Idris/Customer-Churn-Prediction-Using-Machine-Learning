# Customer-Churn-Prediction-Using-Machine-Learning

<img width="1536" height="1024" alt="Image" src="https://github.com/user-attachments/assets/6fe4bf72-186d-4d06-8216-5dce09c4a129" />


## Project Overview

Predicting customer churn is one of the most pressing challenges for subscription‑based businesses such as telecommunications, internet service providers, and digital platforms. Losing customers unnecessarily leads to revenue decline and increased acquisition costs.

This project builds an end‑to‑end machine learning solution **to predict whether a customer will churn** based on their demographic, service usage, billing, and support interaction data. It follows a complete data science workflow from **data exploration and model development to deployment strategy**.

## Business Problem Statement

Customer churn results in significant revenue loss and poor business performance when not managed effectively. A telecommunications company wants to:

* Identify customers at high risk of leaving

* Understand which factors most influence churn

* Prioritize customer retention efforts

* Improve contract and service strategies

This solution simulates a real‑life decision‑making tool that can be integrated into business operations to support **proactive customer retention**.

## Dataset Description

The dataset used for this project contains **1,500 customer records** with features including demographics, tenure, services, billing, support, and churn status.

| Feature | Feature Description |
| :--- | :--- |
| **Age** | Age of the customer |
| **Gender** | Male / Female |
| **Tenure_Months** | Number of months with the company |
| **Monthly_Charges** | Monthly billing amount |
| **Contract_Type** | Contract duration (Month-to-month, One year, Two year) |
| **Internet_Service** | Type of internet service (DSL / Fiber / None) |
| **Support_Calls** | Number of support calls |
| **Payment_Method** | Payment method (Card / Bank Transfer / Cash) |
| **Total_Charges** | Total lifetime charges |
| **Churn** | Target variable (Yes / No) |

## Tools & Technologies

* Python

* Pandas, NumPy

* Matplotlib, Seaborn

* Scikit‑Learn

* Streamlit (for deployment strategy)

* Jupyter Notebook

## Exploratory Data Analysis (EDA)
 
Exploratory Data Analysis (EDA) was conducted to understand customer behavior, assess data quality, and identify patterns associated with churn.
It helped validate business assumptions and highlight key drivers such as contract type, tenure, and billing behavior.
These insights guided feature selection and ensured the machine learning model was both accurate and interpretable.

## Data Preprocessing

* **Encoding:** One‑hot encoding for categorical features

* **Scaling:** Standard scaling for numerical features

* **Pipeline:** Built using Scikit‑Learn ColumnTransformer and Pipeline

* **Train/Test Split:** Stratified split (80/20)

## Model Development

### **Baseline Model:** Logistic Regression
* Simple, interpretable model
* Assesses linear relationships

### Advanced Model: Random Forest Classifier
* Captures non‑linear patterns
* Provides feature importance

## Model Evaluation
### Model Performance Comparison

| Model | Class | Precision | Recall | F1-Score | Overall Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | No Churn | 0.83 | 0.88 | 0.85 | 0.77 |
| | Churn | 0.55 | 0.45 | 0.50 | |
| **Random Forest** | No Churn | **0.84** | **0.93** | **0.88** | **0.82** |
| | Churn | **0.70** | **0.46** | **0.56** | |

Random Forest performed better in identifying churned customers while maintaining balanced precision and recall.

## Key Findings & Business Insights

* **Contract Type:** Month‑to‑month customers are at highest risk of churn.

* **Tenure:** Early tenure customers churn more often.

* **Support Calls:** High support interaction indicates dissatisfaction.

* **Billing:** Higher monthly and total charges correlate with churn.

These insights inform targeted retention strategies and contract incentives.

## Model Deployment (Streamlit App)

To demonstrate real-world usability, the trained churn prediction model was deployed as an interactive Streamlit web application.

### App Capabilities:

* Accepts customer demographic, contract, and billing inputs through a user-friendly interface

* Predicts customer churn probability in real time and classifies risk levels

* Demonstrates how machine learning models can be converted into practical decision-support tools for customer retention

