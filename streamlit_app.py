import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

rating = ["Very Poor","Poor","Fair","Below Average","Average","Above Average",
           "Good","Very Good","Excellent","Very Excellent"]

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
           
    overallQuality = st.slider("Rates the overall material and finish of the house", 1, 10, 5)
    st.write("The overall material and finish of the house is : ", rating[overallQuality - 1])

    # Correctly indented date input
    yearBuilt = st.date_input("Original construction date", datetime.date(2019, 7, 6))
    st.write("The original construction date is: ", yearBuilt)
           
    yearRemodAdd = st.date_input("Remodel date", datetime.date(2019, 7, 6))
    st.write("The remodel date date is: ", yearBuilt)

    totalBasmtSF = st.slider("Total square feet of basement area", 0, 10000, 500)
    st.write("Total square feet of basement area is : ", totalBasmtSF, "sqft")

    floorSF = st.slider("First Floor square feet", 0, 10000, 500)
    st.write("First Floor square feet is : ", floorSF, "sqft")

    grLiveArea = st.slider("Above grade (ground) living area square feet", 0, 10000, 500)
    st.write("Above grade (ground) living area square feet is : ", grLiveArea, "sqft")

    fullBath = st.slider("Full bathrooms above grade", 0, 10, 5)
    st.write("Full bathrooms above grade is : ", fullBath)

    kitchenQual = st.selectbox('Kitchen Quality', ('Excellent','Good','Average','Fair','Poor'))

    garageCars = st.slider("Size of garage in car capacity", 0, 10, 3)
    st.write("Size of garage in car capacity is : ", grLiveArea)

    saleCondition = st.selectbox('Condition of Sale', ('Normal Sale','Abnormal Sale','Adjoining Land Purchase',
                                                      'Allocation','Family','Partial'))
























