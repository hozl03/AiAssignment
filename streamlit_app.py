import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import sklearn # import the module so you can use it
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures

import joblib

loaded_random_forest = joblib.load('random_forest_model.joblib')
loaded_svr = joblib.load('svr_model.joblib')
loaded_lin_reg = joblib.load('linear_regression_model.joblib')




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
           

    # Select more important columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Numeric columns only
    if 'SalePrice' in numeric_df.columns:
        important_num_cols = list(numeric_df.corr()["SalePrice"][
            (numeric_df.corr()["SalePrice"] > 0.50) | (numeric_df.corr()["SalePrice"] < -0.50)
        ].index)
    else:
        important_num_cols = []

    # Categorical columns
    cat_cols = ["MSZoning", "Utilities", "BldgType", "KitchenQual", "SaleCondition", "LandSlope"]
    important_cols = important_num_cols + cat_cols

    df_filtered = df[important_cols]
    st.write("**Filtered Data with Important Columns**")

    df_filtered = df_filtered.drop('GarageArea', axis=1)
    st.write(df_filtered)


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

    # Map kitchen quality input to corresponding code
    kitchenQual = st.selectbox('Kitchen Quality', list(kitchenQual_mapping.keys()))
    kitchenQual_code = kitchenQual_mapping[kitchenQual]  # Map to the corresponding code (e.g., "Ex", "Gd")
    st.write("Kitchen Quality code selected is: ", kitchenQual_code)

    garageCars = st.slider("Size of garage in car capacity", 0, 10, 3)
    st.write("Size of garage in car capacity is : ", grLiveArea)

    # Sale Condition input with mapping
    saleCondition = st.selectbox('Condition of Sale', list(saleCondition_mapping.keys()))
    saleCondition_code = saleCondition_mapping[saleCondition]
    st.write("Sale Condition code selected is: ", saleCondition_code)

    # Corrected data dictionary with valid variable names
    data = {
           'MSZoning': msZoning_code,  # Use msZoning_code
           'Utilities': utility_code,  # Use utility_code
           'LandSlope': landSlope_code,  # Use landSlope_code
           'BldgType': buildingType_code,  # Use buildingType_code
           'OverallQual': overallQuality,  # Use overallQuality from slider
           'YearBuilt': yearBuilt.year,  # Extract the year from date input
           'YearRemodAdd': yearRemodAdd.year,  # Extract the year from date input
           'TotalBsmtSF': totalBasmtSF,  # Use totalBasmtSF from slider
           'TotRmsAbvGrd': totalRmsAbvGrd,  # Use totalRmsAbvGrd from slider
           '1stFlrSF': floorSF,  # Use floorSF from slider
           'GrLivArea': grLiveArea,  # Use grLiveArea from slider
           'FullBath': fullBath,  # Use fullBath from slider
           'KitchenQual': kitchenQual_code,  # Use kitchenQual_code from selectbox
           'GarageCars': garageCars,  # Use garageCars from slider
           'SaleCondition': saleCondition_code,  # Use saleCondition_code from selectbox
    }



# Ensure input_df has the same structure as df_filtered (used in training)
input_df = pd.DataFrame(data, index=[0])
st.write(input_df)
input_data = pd.concat([input_df, df_filtered], axis=0)

important_num_cols.remove("GarageArea")
# Handle categorical variables before numeric scaling
X = pd.get_dummies(input_data, columns=cat_cols)

st.write(X)

# Ensure SalePrice is not in important_num_cols
if 'SalePrice' in important_num_cols:
    important_num_cols.remove("SalePrice")

# Handle the case where the important numeric columns are scaled after dummy encoding
# Check if important_num_cols exist in X
missing_cols = [col for col in important_num_cols if col not in X.columns]

if missing_cols:
    st.write(f"Warning: The following important numeric columns are missing from the dataset after processing: {missing_cols}")

# Standardization of data
scaler = StandardScaler()
# Apply scaler only on numeric columns
X[important_num_cols] = scaler.fit_transform(X[important_num_cols])

# Model selection and prediction
model_choice = st.selectbox('Select Model', ['Random Forest', 'SVR', 'Linear Regression'])

# Prediction using different models
st.write("## Prediction Results")
if st.button('Predict'):
    # Linear Regression prediction
    lin_reg_pred = loaded_lin_reg.predict(input_data)
    st.write(f"**Linear Regression Prediction: ${lin_reg_pred[0]:,.2f}**")

    # Support Vector Regressor prediction
    svr_pred = loaded_svr.predict(input_data)
    st.write(f"**SVR (GridSearch) Prediction: ${svr_pred[0]:,.2f}**")

    # Random Forest Regressor prediction
    random_forest_pred = loaded_random_forest.predict(input_data)
    st.write(f"**Random Forest Prediction: ${random_forest_pred[0]:,.2f}**")



# input_df = pd.get_dummies(input_df, columns=cat_cols)

# # Ensure input_df has the same structure as df_filtered (used in training)
# input_df = pd.get_dummies(input_df, columns=cat_cols)
# input_df = input_df.reindex(columns=df_filtered_drop.columns, fill_value=0)
# input_df = input_df.fillna(0)  # Fill missing values


# # Making prediction
# if model_choice == 'Random Forest':
#     prediction = loaded_random_forest.predict(input_df)
# elif model_choice == 'SVR':
#     prediction = loaded_svr.predict(input_df)
# else:
#     prediction = loaded_lin_reg.predict(input_df)

# # Display Prediction
# st.subheader(f'Predicted House Price: ${prediction[0]:,.2f}')


