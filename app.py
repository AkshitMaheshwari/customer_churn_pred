import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')


# Function to scale the input features
def scale_input_data(input_data, scaler):
    # Scaling the input data using the same scaler used during training
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled


df = pd.read_csv('Churn_Modelling.csv')
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
df['Geography_Germany'] = (df['Geography'] == 'Germany').astype(int)
df['Geography_Spain'] = (df['Geography'] == 'Spain').astype(int)
df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)
df = df.drop(columns=['Geography', 'Gender'])

scaler = StandardScaler()
scaler.fit(df)
# Streamlit app
st.title("Customer Churn Prediction App")
st.write("Enter customer details below to predict the likelihood of churn:")

# User input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=619)
age = st.number_input("Age", min_value=18, max_value=100, value=42)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=2)
balance = st.number_input("Balance ($)", min_value=0.0, max_value=1e6, value=0.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_credit_card = st.selectbox("Has Credit Card?", ['Yes', 'No'])
is_active_member = st.selectbox("Is Active Member?", ['Yes', 'No'])
estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=1e6, value=101348.88)
geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
gender = st.selectbox("Gender", ['Male', 'Female'])

geography_germany = 1 if geography == 'Germany' else 0
geography_spain = 1 if geography == 'Spain' else 0
gender_male = 1 if gender == 'Male' else 0
has_credit_card = 1 if has_credit_card == 'Yes' else 0
is_active_member = 1 if is_active_member == 'Yes' else 0


# Transform categorical variables to numerical
# has_credit_card = 1 if has_credit_card == 'Yes' else 0
# is_active_member = 1 if is_active_member == 'Yes' else 0

# Combine all input data into a DataFrame
input_data = pd.DataFrame([[credit_score, age, tenure, balance, num_of_products, has_credit_card, is_active_member,
                            estimated_salary, geography_germany, geography_spain, gender_male]],
                          columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                                   'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain',
                                   'Gender_Male'])

# Scale the input data
input_data_scaled = scale_input_data(input_data, scaler)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]  # Get the probability value
    churn_result = (churn_probability > 0.5).astype(int)
    churn_not = 1.00 - churn_probability

    if churn_result == 1:
        st.write(f"The customer is likely to churn with a probability of {churn_probability:.2f}.")
    else:
        st.write(f"The customer is not likely to churn with a probability of {churn_not:.2f}.")
    st.write(f"Prediction Probability: {churn_probability}")
