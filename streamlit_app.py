import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

rating = ["Very Poor","Poor","Fair","Below Average","Average","Above Average",
           "Good","Very Good","Excellent","Very Excellent"]

# Mapping for MSZoning
msZoning_mapping = {
    'Agriculture': 'A',
    'Commercial': 'C',
    'Floating Village Residential': 'FV',
    'Industrial': 'I',
    'Residential High Density': 'RH',
    'Residential Low Density': 'RL',
    'Residential Low Density Park ': 'RP',
    'Residential Medium Density': 'RM'
}

utility_mapping = {
    'All Public Utilities': 'AllPub',
    'Electricity, Gas, and Water (Septic Tank)': 'NoSewr',
    'Electricity and Gas Only': 'NoSeWa',
    'Electricity Only': 'ELO'
}

landSlope_mapping = {
    'Gentle slope': 'Gtl',
    'Moderate Slope': 'Mod',
    'Severe Slope': 'Sev'
}

buildingType_mapping = {
    'Single-family Detached': '1Fam',
    'Two-family Conversion': '2FmCon',
    'Duplex': 'Duplx',
    'Townhouse End Unit': 'TwnhsE',
    'Townhouse Inside Unit': 'TwnhsI'
}

kitchenQual_mapping = {
    'Excellent': 'Ex',
    'Good': 'Gd',
    'Average': 'TA',
    'Fair': 'Fa',
    'Poor': 'Po'
}

saleCondition_mapping = {
    'Normal Sale': 'Normal',
    'Abnormal Sale': 'Abnorml',
    'Adjoining Land Purchase': 'AdjLand',
    'Allocation': 'Alloca',
    'Family': 'Family',
    'Partial': 'Partial'
}


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
           
    msZoning = st.selectbox('Zoning', list(msZoning_mapping.keys()))
    msZoning_code = msZoning_mapping[msZoning]  # Map to the corresponding code (e.g., "A", "C")
    st.write("Zoning code selected is: ", msZoning_code)

    
    # Utility input with mapping
    utility = st.selectbox('Utility', list(utility_mapping.keys()))
    utility_code = utility_mapping[utility]
    st.write("Utility code selected is: ", utility_code)

    # Land Slope input with mapping
    landSlope = st.selectbox('Land Slope', list(landSlope_mapping.keys()))
    landSlope_code = landSlope_mapping[landSlope]
    st.write("Land Slope code selected is: ", landSlope_code)

    # Building Type input with mapping
    buildingType = st.selectbox('Building Type', list(buildingType_mapping.keys()))
    buildingType_code = buildingType_mapping[buildingType]
    st.write("Building Type code selected is: ", buildingType_code)
           
    overallQuality = st.slider("Rates the overall material and finish of the house", 1, 10, 5)
    st.write("The overall material and finish of the house is : ", rating[overallQuality - 1])

    # Correctly indented date input
    yearBuilt = st.date_input("Original construction date", datetime.date(2019, 7, 6))
    st.write("The original construction date is: ", yearBuilt)
           
    yearRemodAdd = st.date_input("Remodel date", datetime.date(2019, 7, 6))
    st.write("The remodel date date is: ", yearBuilt)

    totalBasmtSF = st.slider("Total square feet of basement area", 0.0, 10000.0, 500.0)
    st.write("Total square feet of basement area is : ", totalBasmtSF, "sqft")

    totalRmsAbvGrd = st.slider("Total rooms above grade (does not include bathrooms)", 1, 20, 10)
    st.write("Total rooms above grade (does not include bathrooms) is : ", totalRmsAbvGrd)


    floorSF = st.slider("First Floor square feet", 0.0, 10000.0, 500.0)
    st.write("First Floor square feet is : ", floorSF, "sqft")

    grLiveArea = st.slider("Above grade (ground) living area square feet", 0.0, 10000.0, 500.0)
    st.write("Above grade (ground) living area square feet is : ", grLiveArea, "sqft")

    fullBath = st.slider("Full bathrooms above grade", 0, 10, 5)
    st.write("Full bathrooms above grade is : ", fullBath)

    kitchenQual = st.selectbox('Kitchen Quality', ('Excellent','Good','Average','Fair','Poor'))

    garageCars = st.slider("Size of garage in car capacity", 0, 10, 3)
    st.write("Size of garage in car capacity is : ", grLiveArea)

    # Sale Condition input with mapping
    saleCondition = st.selectbox('Condition of Sale', list(saleCondition_mapping.keys()))
    saleCondition_code = saleCondition_mapping[saleCondition]
    st.write("Sale Condition code selected is: ", saleCondition_code)

    data = {
               'msZoning': MSZoning
               'utility': Utilities
               'landSlope': LandSlope
               'buildingType': BldgType
               'overallQuality': OverallQual
               'yearBuilt': YearBuilt
               'yearRemodAdd': YearRemodAdd
               'totalBasmtSF': TotalBsmtSF
               'totalRmsAbvGrd': TotRmsAbvGrd
               'floorSF': 1stFlrSF
               'grLiveArea': GrLivArea
               'fullBath': FullBath
               'kitchenQual': KitchenQual
               'garageCars' : GarageCars
               'saleCondition': SaleCondition
    }
    input_df = pd.DataFrame(data, index=[0])
    input_house = pd.concat([input_df, X], axis = 0)

input_df






















