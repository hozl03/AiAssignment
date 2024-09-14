import streamlit as st
import pandas as pd

st.title('House Price Prediction')

st.write('This is an app builds a house price prediction machine learning model')


with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('train.csv')
  df

  st.write('**Statistical Summary of Dataset**')
  summary = df.describe().T
  summary

  st.write('**WaiKian is gay**')

