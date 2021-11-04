
# Customer Churn Prediction

##  *A machine learning model  for Customer Churn Prediction*
- [Live Demo](https://share.streamlit.io/tapiwachamb/customer_churn/main/app.py)


![Logo](https://drive.google.com/uc?id=1dvH9bOF-WCEOoIoVLOh67ob388tkzbcW&export=download)


## TAPIWA CHAMBOKO
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://katherinempeterson.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tapiwa-chamboko-327270208/)
[![github](https://img.shields.io/badge/github-1DA1F2?style=for-the-badge&logo=githubr&logoColor=white)](https://github.com/tapiwachamb)


## 🚀 About Me
I'm a full stack developer experienced in deploying artificial intelligence powered apps


## Authors

- [@Tapiwa chamboko](https://github.com/tapiwachamb)


## Installation

Install required packages 

```bash
  pip install streamlit
  pip install pycaret
  pip insatll scikit-learn==0.23.2
  pip install numpy
  pip install seaborn 
  pip install pandas
  pip install matplotlib
  pip install plotly-express
  pip install streamlit-lottie
```
    
## Acknowledgements

 - [Moez Ali](https://towardsdatascience.com/predict-customer-churn-the-right-way-using-pycaret-8ba6541608ac)
 - Predict Customer Churn (the right way) using PyCaret
 


## Datasets
- [Download Customer Churn Datasets here](https://www.kaggle.com/blastchar/telco-customer-churn)
## Data
- Train_data folder contains the data for training the model
- Test_data folder conain tha the data for testing the model 


## Model Notebook
- *Model notebook is in note_book folder*
- run the notebook in jupyter notebook or [google colab](https://colab.research.google.com/)


**Example code**
- import libraries

```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas
%matplotlib inline
import plotly.express as px
```
- load data in dataframe
```bash
df = pd.read_csv("Train_data/Telcoms-Customer-Churn.csv",header = 0)
df.head()
```
- data preparation
```bash
from pycaret.classification import *
s = setup(data, target = 'Churn', ignore_features = ['customerID'])
```
- Adaboost classifier
```bash
AdaBoostClassifier = ("ada", algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=6671)
```
## Deployment

To deploy this project we used streamlit to create Web App
- Run this code below

```bash
  streamlit run app.py 
```


## Usage/Examples
- custom fuction in pycaret

```bash
def calculate_profit(y, y_pred):
    tp = np.where((y_pred==1) & (y==1), (5000-1000), 0)
    fp = np.where((y_pred==1) & (y==0), -1000, 0)
    return np.sum([tp,fp])
    
# add metric to PyCaret
add_metric('profit', 'Profit', calculate_profit)

```


## Deployed Model
- The Deployed model pipeline  is named Cusomer_Churn.pkl
- This pipeline wil be used in the web app
## Appendix

Happy Coding!!!!!!

