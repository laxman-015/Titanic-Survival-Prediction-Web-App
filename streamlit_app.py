import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
model = pickle.load(open("model.pkl", "rb"))
# Title
st.title("Titanic Survival Prediction")

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Data preprocessing
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Features and target
X = df[['Pclass','Sex','Age','SibSp','Parch','Fare']]
y = df['Survived']

# Train model
model = LogisticRegression()
model.fit(X, y)

st.write("Enter passenger details to predict survival")

# User inputs
pclass = st.selectbox("Passenger Class", [1,2,3])
sex = st.selectbox("Sex", ["male","female"])
age = st.slider("Age", 1,80)
sibsp = st.number_input("Siblings/Spouses aboard",0,5)
parch = st.number_input("Parents/Children aboard",0,5)
fare = st.number_input("Fare",0.0,500.0)

# Convert sex
sex = 0 if sex=="male" else 1

# Prediction
if st.button("Predict"):
    prediction = model.predict([[pclass,sex,age,sibsp,parch,fare]])

    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")
