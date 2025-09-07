import streamlit as st
import pandas as pd
import joblib
from pymfe.mfe import MFE
from pathlib import Path

# Load meta-model
meta_model = joblib.load("../models/meta_model.pkl")  # path to your trained meta-model
meta_features_path = Path("../meta_dataset/meta_features.csv")

# Title
st.title("AutoML Meta-Model Demo")
st.write("Upload a dataset to get top recommended models.")

# File upload
uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    X_new, y_new = df.iloc[:, :-1], df.iloc[:, -1]
    
    st.write("Dataset preview:")
    st.dataframe(df.head())
    
    # Extract meta-features
    mfe = MFE(groups=["general", "statistical", "info-theory"], random_state=42)
    mfe.fit(X_new.values, y_new.values)
    ft_names, ft_values = mfe.extract()
    
    new_meta = pd.DataFrame([ft_values], columns=ft_names)
    
    # Match meta-model features
    X_meta = new_meta[meta_model.feature_names_in_]
    
    # Predict recommended algorithms
    recommended = meta_model.predict_proba(X_meta)[0]
    models = meta_model.classes_
    
    results = pd.DataFrame({
        "model": models,
        "probability": recommended
    }).sort_values(by="probability", ascending=False)
    
    st.write("Top recommended models:")
    st.dataframe(results)
    
    st.write("You can now run hyperparameter tuning only on these top models.")
