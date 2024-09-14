import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title('House Price Prediction')

st.write('This is an app that builds a house price prediction machine learning model')

# Load the data
with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('train.csv')
    st.write(df)

    st.write('**Statistical Summary of Dataset**')
    summary = df.describe().T
    st.write(summary)

    st.write('**Wai Kian is gay**')




