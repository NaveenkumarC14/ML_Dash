import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


data = pd.read_csv('https://raw.githubusercontent.com/NaveenkumarC14/Cardiovascular-Disease-EDA-and-Analytics/main/cardio_train.csv')
# separate the data into features and target
#data=data.drop('id',axis=1)
def calculate_age(days):
  days_year = 365.2425
  age = int(days // days_year)
  return age
data['age_new'] = data['age'].apply(lambda x: calculate_age(x))

features=data[['age_new','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
target=data['cardio']


# split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, stratify=target
)

sepal_length = st.sidebar.text_input('Enter age', '')

sepal_width =st.sidebar.text_input('Gender', '')
        

petal_length = st.sidebar.text_input('Enter Height', '')

petal_width =st.sidebar.text_input('Enter Weight', '')
        
petal_width1 = st.sidebar.text_input('Enter ap_hi', '')
petal_width2 = st.sidebar.text_input('Enter ap_lo', '')
        
petal_width3 = st.sidebar.text_input('Enter cholesterol', '')
petal_width4 = st.sidebar.text_input('Enter gluc', '')
petal_width5 = st.sidebar.text_input('Enter smoke', '')
petal_width6 =st.sidebar.text_input('Enter alco', '')
petal_width7 = st.sidebar.text_input('Enter active', '')
        
        
        
       
values = [sepal_length, sepal_width, petal_length, petal_width,petal_length1,petal_length2,petal_length3,petal_length4,petal_length5,petal_length6,petal_length7]

        
if st.sidebar.button("Predict"):
  values
          
  
