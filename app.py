#streamlit run app.py
import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd

st.title("Car Price Predictor")

car = pd.read_csv("final_data.csv")

# Sidebar start
st.sidebar.header("Enter Car Details")

company = st.sidebar.selectbox(
    "Select company",
    sorted(car["company"].unique())
)
name = st.sidebar.selectbox(
    "Select name",
    sorted(car["name"].unique())
)
year = st.sidebar.number_input(
    "Enter year",
     min_value=2000,
     max_value=2024, 
     step=1
)
kms_driven = st.sidebar.number_input(
    "Enter kilometers driven", 
     min_value=10000, 
     max_value=400000, 
     step = 5000
)
fuel_type = st.sidebar.selectbox(
    "Select fuel type", 
    ["Petrol", "Diesel", "LPG"])

# Main button
if st.sidebar.button("Predict Price"):
   model = pkl.load(open("pipe.pickle", "rb"))
   df = pd.DataFrame([[company, name, year, kms_driven, fuel_type]],
   columns = ["company", "name", "year", "kms_driven", "fuel_type"])
   # df = pd.DataFrame(data, columns=columns)
   #st.write(df)
   result = model.predict(df) 

# Output Below title(Main Page)
   st.write("### Predicting for")
   st.write(f"Company: {company}")
   st.write(f"Name: {name}")
   st.write(f"Year: {year}")
   st.write(f"kms_driven: {kms_driven}")
   st.write(f"fuel_type: {fuel_type}")
    
   st.write("Predicted price: ₹",round(result[0,0]))
