# app.py
import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import requests
from io import BytesIO

# Load the trained RandomForest model
with open("Walmart_sales_forecasting.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Walmart Sales Forecasting", layout="wide")

# Display image
image_url = "https://raw.githubusercontent.com/Masterx-AI/Project_Retail_Analysis_with_Walmart/main/Wallmart1.jpg"
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
st.image(img, use_column_width=True)

st.title("🛒 Walmart Weekly Sales Prediction")

# Sidebar - user inputs
st.sidebar.header("Input Features")

def user_input_features():
    Week = st.sidebar.slider("Week of Year", 1, 52, 1)
    CPI = st.sidebar.number_input("CPI", 200.0, 300.0, 220.0)
    Unemployment = st.sidebar.number_input("Unemployment Rate (%)", 3.0, 15.0, 6.0)
    Size = st.sidebar.number_input("Store Size (sq.ft.)", 5000, 250000, 100000)
    Type = st.sidebar.selectbox("Store Type", ["A", "B", "C"])
    Dept = st.sidebar.number_input("Department Number", 1, 100, 1)
    Store = st.sidebar.number_input("Store Number", 1, 50, 1)

    data = {
        "Week": Week,
        "CPI": CPI,
        "Unemployment": Unemployment,
        "Size": Size,
        "Type": Type,
        "Dept": Dept,
        "Store": Store
    }
    features_df = pd.DataFrame(data, index=[0])
    return features_df

input_df = user_input_features()

# Preprocessing - encode categorical variable
input_df["Type"] = input_df["Type"].map({"A": 0, "B": 1, "C": 2})

# Show input dataframe
st.subheader("User Input Features")
st.write(input_df)

# Predict only when user clicks button
if st.button("Predict Weekly Sales"):
    prediction = model.predict(input_df)
    st.subheader("Predicted Weekly Sales")
    st.write(f"${prediction[0]:,.2f}")
