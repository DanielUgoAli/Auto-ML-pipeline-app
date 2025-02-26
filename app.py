import streamlit as st
import pandas as pd 
import os

# Profiling
from ydata_profiling import ProfileReport
from utils import auto_classify, auto_regression

with st.sidebar:
    # st.image("image.png")
    st.title("AutoML App")
    choice = st.radio("Select Option", ["Upload", "Profiling", "ML", "Download"])
    st.info("This is an application that builds a complete automated ML pipeline workflow using Streamlit, Ydata-profiling and Pycaret.")


if os.path.exists("data.csv"):
    df = pd.read_csv("data.csv")
else:
    df = pd.DataFrame()

if choice == "Upload":
    st.title("Upload Dataset for Modelling")
    file = st.file_uploader("Upload Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("data.csv", index=False)
        st.success("Data Uploaded Successfully")
        st.dataframe(df)


if choice == "Profiling":
    st.title("Automated EDA")
    st.info("EDA - Exploratory Data Analysis")
    if st.button("Generate Report"):
        profile = ProfileReport(df, title="Pandas Profiling Report")
        profile.to_file("output.html")
        st.success("Profile Report Generated")
        st.html("output.html")


if choice == "ML":
    st.title("Automated Machine Learning")
    choice = st.selectbox("Choose the ML Task", ["Classification", "Regression"])

    if choice == "Classification":
        chosen_target = st.selectbox("Choose the Target Column", df.columns)
        if st.button("Run ML eperiment"):
            setup_df, compare_df = auto_classify(data=df, target=chosen_target)
            st.dataframe(setup_df)
            st.dataframe(compare_df)
            st.success("ML Experiment Completed")
            st.info("⬇️This will delete the data.csv file⬇️")
            if st.button("Save space"):
                os.remove("data.csv")
    
    if choice == "Regression":
        chosen_target = st.selectbox("Choose the Target Column", df.columns)
        if st.button("Run ML eperiment"):
            setup_df, compare_df = auto_regression(data=df, target=chosen_target)
            st.dataframe(setup_df)
            st.dataframe(compare_df)
            st.success("ML Experiment Completed")
            st.info("⬇️This will delete the data.csv file⬇️")
            if st.button("Save space"):
                os.remove("data.csv")



if choice == "Download":
    with open("_model.pkl", "rb") as f:
        st.download_button("Download Model", f, file_name="best_model.pkl")


