# pip install openpyxl
import pandas as pd
import streamlit as st
import zipfile
import base64
import os
from PIL import Image
from streamlit_lottie import st_lottie
import json
import requests





def app():
    st.title('Home')
    st.markdown("""
                ## Customer Churn Machine learning App By Tapiwa Chamboko.
                



---
""")

    st.write('')

    st.write('Welcome to Customer Churn prediction App')
    st.image('3.png', width=800)

  
