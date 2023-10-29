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

def app():
    model = load_model('Cusomer_Churn')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    def predict(model, input_df):
        predictions_df = predict_model(estimator=model, data=input_df)
        predictions = predictions_df['prediction_label'][0]
        return predictions
        
    add_selectbox1 = st.sidebar.selectbox(
    "Select your graph?",
    ("Bar", "Area", "Histogram","Scatter","Pie"))
    st.title("Customer Churn Prediction (Batch)")
    
    uploaded_file = st.file_uploader("Upload your input XLSX file", type=["xlsx"])

    if uploaded_file is not None:

        data =  pd.read_excel(uploaded_file,engine='openpyxl')
        predictions = predict_model(estimator=model,data=data)
        #st.write(predictions)
        df = st.dataframe(predictions)

        groupby_column = st.selectbox(
        'What would you like to analyse?',predictions.columns.values.tolist(),)
        output_columns = 'prediction_label'
        tapiwa = predictions.groupby(by=[groupby_column], as_index=False)[output_columns].count()
            
        if add_selectbox1 == 'Bar':

            fig = px.bar(
                
                tapiwa,
                x=groupby_column,
                y='prediction_label',
                color='prediction_label',
                color_continuous_scale=['red', 'yellow', 'green','blue'],
                template='plotly_white',
                title=f'<b>Predictions by {groupby_column}</b>'
                )
            st.plotly_chart(fig)

        if add_selectbox1 == 'Area':

            fig2 = px.area(
                tapiwa,
                x=groupby_column,
                y='prediction_label',
                color='prediction_label',
                template='plotly_white',
                title=f'<b>Prodictions by {groupby_column}</b>'
                )
            st.plotly_chart(fig2)

        if add_selectbox1 == 'Histogram':
            tapiwa.hist()
            plt.show()
            st.pyplot()

        if add_selectbox1 == 'Scatter':
            fig3 = px.scatter(
                x=predictions["prediction_label"],
                y=predictions[groupby_column],
            )
            fig3.update_layout(
                xaxis_title="Predictions",
                yaxis_title=f'<b>{groupby_column}</b>',
        )
            st.write(fig3)

        if add_selectbox1 == 'Pie':
            fig4 = px.pie(predictions, values=predictions['prediction_label'], names=predictions[groupby_column])
            st.plotly_chart(fig4)

            

        dfs= pd.DataFrame(predictions)
        towrite = io.BytesIO()
        downloaded_file = dfs.to_excel(towrite, encoding='utf-8', index=False, header=True)
        towrite.seek(0)  # reset pointer
        b64 = base64.b64encode(towrite.read()).decode()  # some strings
        linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="myfilename.xlsx">Download excel file</a>'
        st.markdown(linko, unsafe_allow_html=True)

    else:
        st.write("")
        st.write("Drop the excel file in the Test_data folder")
        st.info('Awaiting for Exel File.')

        st.image('8.png', width=700) 
