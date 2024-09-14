import streamlit as st
import pandas as pd

st.title('House Price Prediction')

st.write('This is an app builds a house price prediction machine learning model')

df = pd.read_csv('train.csv')
df
