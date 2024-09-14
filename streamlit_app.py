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

Data Visualization
with st.expander('Data Visualization'):
    st.write('**Scatter Plot**')
    st.scatter_chart(data=df, x='OverallQual', y='SalePrice')  # Modify the x and y axis as per your dataset

    st.write('**Correlation Heatmap**')
    # Generate heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="RdBu", ax=ax)
    ax.set_title("Correlations Between Variables", size=15)
    st.pyplot(fig)
