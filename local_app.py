import pandas as pd
import streamlit as st
import numpy as np 
import joblib
from sklearn.svm import SVC, LinearSVC

st.title("IS YOUR WATER POTABLE?")

st.write(""" 
### Enter values of all the features of the water to determine if it is potable?
""")

#df = pd.read_csv('./cleaned.csv')
    


col1, col2, col3 = st.columns(3)

ph = col2.number_input("Ph value")
Hardness = col3.number_input("Hardness value in mg/L")
Solids = col1.number_input("TDS(Total dissolved solids) in ppm")
Chloramines = col2.number_input("Chloramines value in ppm")
Sulfate = col3.number_input("Sulfate value in mg/L")
Conductivity = col1.number_input("Conductivity value μS/cm")
Organic_carbon = col2.number_input("Organic_carbon value in ppm")
Trihalomethanes = col3.number_input("Trihalomethanes value in μg/L")
Turbidity = col1.number_input("Turbidity(light emitting property) of water in NTU")


df_pred = pd.DataFrame([[ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity]],
columns= ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity'])

model = joblib.load('./fhs_svm_model.pkl')
prediction = model.predict(df_pred)

if st.button('PREDICT'):
    if(prediction[0]==0):
        st.write('<p class="big-font">Water is not potable and is not recommended for human consumption.</p>',unsafe_allow_html=True)

    else:
        st.write('<p class="big-font">Water is potable and  fit for consumption.</p>',unsafe_allow_html=True)