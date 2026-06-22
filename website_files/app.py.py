import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load Models
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
item_similarity_df = pd.read_csv('item_similarity.csv', index_col=0)

st.title("🛒 Shopper Spectrum Dashboard")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Product Recommendation", "Customer Segmentation"])

if menu == "Product Recommendation":
    st.header("Product Recommender")
    product_list = item_similarity_df.columns.tolist()
    product_name = st.selectbox("Select or Enter a Product Name", product_list)

    if st.button("Recommend"):
        recommendations = item_similarity_df[product_name].sort_values(ascending=False)[1:6].index.tolist()
        st.subheader("Recommended Products:")
        for item in recommendations:
            st.write(f"- {item}")

elif menu == "Customer Segmentation":
    st.header("Customer Segmentation Module")
    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=10)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0)

    if st.button("Predict Segment"):
        # Scale the inputs
        input_scaled = scaler.transform([[recency, frequency, monetary]])
        # Predict
        cluster = kmeans.predict(input_scaled)[0]

        # Map cluster to label (based on our earlier logic)
        labels = {0: "High-Value", 1: "Regular", 2: "Occasional", 3: "At-Risk"}
        st.success(f"This customer belongs to: **{labels.get(cluster, 'Unknown')} Shopper**")
