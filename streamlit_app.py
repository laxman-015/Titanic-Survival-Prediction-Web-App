import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("Titanic Survival Prediction")

st.write("Enter passenger details to predict survival")

# User inputs
pclass = st.selectbox("Passenger Class", [1,2,3])
age = st.slider("Age", 1,80)
sibsp = st.number_input("Siblings/Spouses aboard",0,5)
parch = st.number_input("Parents/Children aboard",0,5)
fare = st.number_input("Fare",0.0,500.0)
sex = st.selectbox("Sex",["male","female"])

# Convert input
sex = 0 if sex=="male" else 1

# Prediction button
if st.button("Predict"):
    
    input_data = [[pclass,sex,age,sibsp,parch,fare]]
    
    prediction = model.predict(input_data)
    
    if prediction[0]==1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")