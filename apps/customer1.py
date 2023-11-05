from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import plotly.express as px
import base64
import os
import io
from PIL import Image
#customer lifetime value.
#business-smart metric,taking into account the risk and reward proposition.
def app():
    model = load_model('Cusomer_Churn')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    def predict(model, input_df):
        predictions_df = predict_model(estimator=model, data=input_df)
        predictions = predictions_df['prediction_label'][0]
        return predictions

   
    st.title("Customer Churn Prediction")
    st.sidebar.markdown("""# Select features

""")
    #{'apple':1, 'bat':2, 'car':3, 'pet':4}
    gender = st.sidebar.selectbox('gender', ['Male', 'Female'])
    SeniorCitizen = st.sidebar.number_input('SeniorCitizen', 0,1)
    tenure = st.sidebar.number_input('tenure', min_value=0, max_value=80, value=0)
    MultipleLines = st.sidebar.selectbox('MultipleLines', ['Yes', 'No', 'No phone service'])
        
    InternetService = st.sidebar.selectbox('InternetService', ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.sidebar.selectbox('OnlineSecurity', ['Yes', 'No', 'No phone service'])
    OnlineBackup = st.sidebar.selectbox('OnlineBackup', ['Yes', 'No', 'No phone service'])
    DeviceProtection = st.sidebar.selectbox('DeviceProtection', ['Yes', 'No', 'No phone service'])

    TechSupport = st.sidebar.selectbox('TechSupport', ['Yes', 'No', 'No phone service'])
    StreamingTV = st.sidebar.selectbox('StreamingTV', ['Yes', 'No', 'No phone service'])
    StreamingMovies = st.sidebar.selectbox('StreamingMovies', ['Yes', 'No', 'No phone service'])
    Contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        
    PaymentMethod = st.sidebar.selectbox('PaymentMethod', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'])
    MonthlyCharges = st.sidebar.number_input('MonthlyCharges', min_value=18.00, max_value=120.00, step=0.10)
    TotalCharges = st.sidebar.number_input('TotalCharges', min_value=18.00, max_value=8700.00, step=1.00)
    if st.sidebar.checkbox('Partner'):
        Partner = 'yes'
    else:
        Partner = 'no'

    if st.sidebar.checkbox('Dependents'):
        Dependents = 'yes'
    else:
        Dependents = 'no'

    if st.sidebar.checkbox('PhoneService'):
        PhoneService = 'yes'
    else:
        PhoneService = 'no'
    if st.sidebar.checkbox('PaperlessBilling'):
        PaperlessBilling = 'yes'
    else:
        PaperlessBilling = 'no'
        

    output=""
    action=""

    input_dict = {'gender' : gender, 'SeniorCitizen' : SeniorCitizen, 'tenure' : tenure, 'MultipleLines' : MultipleLines, 'InternetService' : InternetService, 'OnlineSecurity' : OnlineSecurity
    ,'OnlineBackup' : OnlineBackup, 'DeviceProtection' : DeviceProtection, 'TechSupport' : TechSupport, 'StreamingTV' : StreamingTV, 'StreamingMovies' : StreamingMovies, 'Contract' : Contract,
    'PaymentMethod' : PaymentMethod, 'MonthlyCharges' : MonthlyCharges, 'TotalCharges' : TotalCharges, 'Partner' : Partner, 'Dependents' : Dependents, 'PhoneService' : PhoneService, 'PaperlessBilling' : PaperlessBilling}
    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        output = predict(model=model, input_df=input_df)
            
        if output=="Yes":
            output = 'Arlet ! This customer has a high chance of leaving the business.' + '  ' +'The predicted outcome is :'+' '+str(output)
            action="Contact Risk !"
            st.error(output + '  ' + action)
        else:
            output = 'Great ! This customer is loyal to the business.' + '  ' +'The predicted outcome is :'+' '+str(output)
            st.success(output + '  ' + action)

    else:
        st.write("")
        st.info('Awaiting for The prediction.')
        st.image('10.png', width=700)
