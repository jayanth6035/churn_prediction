#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# read the excel file
df = pd.read_excel(r"C:\Users\priya\Downloads\customer_churn_large_dataset.xlsx")

# Encode Categorical Variables
# Let's use Label Encoding for simplicity (replace with one-hot encoding if needed)
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Location']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])
    
df.drop(['CustomerID','Name'], inplace=True, axis=1)

# Split the data into features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# splitting of traing testing set into X and y
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standarization
scaler =StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title('Churn prediction')
st.write('This app predicts Churn  based on the input features.')

# Sidebar inputs
st.sidebar.header('Input Features')
age = st.sidebar.slider('Age', min_value=0, max_value=100)
gender = st.sidebar.slider('Gender', min_value=0, max_value=1)
location = st.sidebar.slider('Location', min_value=0, max_value=4)
subscription_length_months = st.sidebar.slider('Subscription_Length_Months', min_value=0, max_value=36)
monthly_bill = st.sidebar.slider('Monthly_Bill', min_value=20.00, max_value=150.00)
total_usage_gb = st.sidebar.slider('Total_Usage_GB', min_value=30, max_value=1000)

# Predict button
predict_button = st.sidebar.button('Predict')

# Check if the Predict button is clicked
if predict_button:
    # Create a dataframe with the selected input features
    input_data = pd.DataFrame({
        'age' : [age],
        'gender' : [gender],
        'location' : [location],
        'subscription_length_months' : [subscription_length_months],
        'monthly_bill' : [monthly_bill],
        'total_usage_gb' : [total_usage_gb]
    })
    
    # Standardize the input data using the same scaler
    input_data_scaled = scaler.transform(input_data)
    
     # Predict churn
    churn_prediction = model.predict(input_data_scaled)

    # Display the prediction
    st.subheader('Churn Prediction')
    if churn_prediction[0] == 0:
        st.write("Prediction: Not Churned")
    else:
        st.write("Prediction: Churned")

    
    

