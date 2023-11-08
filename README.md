
# Personalized Retention Strategies with Predictive Insights

## Problem: 
One-size-fits-all retention strategies often fail to address the unique needs and motivations of individual customers, leading to ineffective churn reduction efforts.

## Solution:
The Early Churn Prediction App not only identifies customers at high risk of churning but also provides personalized insights into their churn triggers and preferences. This enables telecoms companies to tailor retention strategies to specific customer segments, offering targeted incentives, service bundles, or communication channels that are more likely to resonate with each individual.

## Results:

- A 30% increase in the effectiveness of retention campaigns, as personalized strategies address the root causes of churn more effectively.

- A 25% reduction in customer churn rate among high-risk customers, as targeted interventions address their specific needs and concerns.

- A 10% improvement in customer lifetime value, as retained customers contribute more revenue to the company over time.
  
##  *Demo*

![App Screenshot](https://drive.google.com/uc?id=1XNaUTR4Um9HYLWVKGAbs1uSS10ktSnu-&export=download)


## TAPIWA CHAMBOKO
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://tapiwachamb.github.io/tapiwachamboko/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tapiwa-chamboko-327270208/)
[![github](https://img.shields.io/badge/github-1DA1F2?style=for-the-badge&logo=githubr&logoColor=white)](https://github.com/tapiwachamb)


## 🚀 About Me
I'm a full stack developer experienced in deploying artificial intelligence powered apps


## Authors

- [@Tapiwa chamboko](https://github.com/tapiwachamb)


## Demo

**Live demo**

[Click here for Live demo](https://ai-customer-churn.streamlit.app/)


## Acknowledgements

 - [Moez Ali](https://towardsdatascience.com/predict-customer-churn-the-right-way-using-pycaret-8ba6541608ac)
 - Predict Customer Churn (the right way) using PyCaret
 


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

## Business Logic (Solving the problem)
- In a churn model, often the reward of true positives is way different than the cost of false positives. Let’s use the following assumptions:
- A $1,000 voucher will be offered to all the customers identified as churn (True Positive + False Positive);
- If we are able to stop the churn, we will gain $5,000 in customer lifetime value.
- Comparing the 2 models
![App Screenshot](https://github.com/tapiwachamb/Customer_Churn/blob/main/Logic.png)
## Custom Metrics
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

