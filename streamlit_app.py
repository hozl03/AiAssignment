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

with st.expander('Data Visualization'):
    st.write('**Scatter Plot**')
    st.scatter_chart(data=df, x='OverallQual', y='SalePrice')  # Modify as needed

    st.write('**Correlation Heatmap**')
    
    # Filter out non-numeric columns for the heatmap
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if not numeric_df.empty:
        # Generate heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), cmap="RdBu", ax=ax)
        ax.set_title("Correlations Between Variables", size=15)
        st.pyplot(fig)
    else:
        st.write("No numeric columns available for correlation heatmap.")



# Input
with st.sidebar:
    st.header('Input features')
    msZoning = st.selectbox('Zoning', ('Agriculture', 'Commercial', 'Floating Village Residential','Industrial', 
                                      'Residential High Density','Residential Low Density','Residential Low Density Park ',
                                      'Residential Medium Density'))
    
    utility = st.selectbox('Utility', ('Electricity, Gas, and Water', 'Electricity and Gas Only', 'Electricity only',
                                      'All Public Utilities'))
    
    landSlope = st.selectbox('Land Slope', ('Gentle slope', 'Moderate Slope', 'Severe Slope'))

    buildingType = st.selectbox('Building Type', ('Single-family Detached', 'Two-family Conversion',
                                                  'Duplex', 'Townhouse End Unit', 'Townhouse Inside Unit'))
    





















