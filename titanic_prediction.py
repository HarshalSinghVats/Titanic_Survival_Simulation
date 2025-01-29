import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

#PAGE CONFIGURATION...
st.set_page_config(
    page_title="Titanic Prediction HSV", 
    layout="wide",                  
    initial_sidebar_state="collapsed")

scaler=joblib.load('titanic_scaler.joblib')

@st.cache_resource

def load_model(file_path):
    return joblib.load(file_path)

titanic_model=joblib.load('titanic_prediction.joblib')

st.title("Titanic Survival Prediction")

pass_class=st.number_input("Passenger's class : ", min_value=1,value=1)
pass_sex=st.selectbox("Passenger's sex : ", ["Male", "Female"])
pass_age=st.number_input("Passenger's age : ", min_value=1,value=18)
pass_sib_spouse=st.number_input("Passenger's siblings/spouse : ", min_value=0,value=1)
pass_par_child=st.number_input("Passenger's parents/children : ", min_value=0,value=2)
pass_fare=st.number_input("Passenger's fare : ",min_value=0.0,value=15.00,format="%.3f")
pass_destination=st.selectbox("Passenger's destination : ", ["Cherbourg","Queenstown","Southampton"])

pass_sex = 1 if pass_sex == "Male" else 0
pass_destination = 1 if pass_destination == "Queenstown" else 0 if pass_destination=="Cherbourg" else 2

if st.button("Predict"):
    input_data=np.array([[pass_class,pass_sex,pass_age,pass_sib_spouse,pass_par_child,pass_fare,pass_destination]])
    input_data=input_data.reshape(1,-1)
    input_data=scaler.transform(input_data)
    prediction = titanic_model.predict(input_data)
    st.write("Prediction:", "Survived" if prediction[0] == 1 else "Not Survived")
