import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures

st.title('House Price Prediction')

st.write('This is an app builds a house price prediction machine learning model')


with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('train.csv')
  df

  st.write('**Statistical Summary of Dataset**')
  summary = df.describe().T
  summary

  st.write('**Who else is gay**')
  plt.figure(figsize=(10,8))
  sns.heatmap(df.corr(), cmap="RdBu")
  plt.title("Correlations Between Variables", size=15)
  plt.show()
