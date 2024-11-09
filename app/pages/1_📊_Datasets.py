import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
import os

st.set_page_config(
    page_title="Dataset Management",
    page_icon="✅",
)

automl = AutoMLSystem.get_instance()

# Path for the datasets folder
DATASETS_PATH = "assets/datasets/"

# Ensure the datasets directory exists
os.makedirs(DATASETS_PATH, exist_ok=True)

st.title("Dataset Management")

# Upload and Convert CSV to Dataset
st.header("Upload and Create Dataset")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    # Read CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset:")
    st.dataframe(df)

    # Give dataset a name
    dataset_name = st.text_input("Dataset Name", "default_dataset")

    # Button to convert and save dataset
    if st.button("Convert and save the dataset"):
        # Define asset path 
        asset_path = f"assets/datasets/{dataset_name}.csv"

        # Convert DataFrame to Dataset
        dataset = Dataset.from_dataframe(data=df, name=dataset_name, asset_path=asset_path)

        # Use the AutoMLSystem artifact registry to save the dataset
        automl.registry.save(dataset)

        # When successful 
        st.success(f"Dataset '{dataset_name}' has been successfully saved as an artifact.")

datasets = automl.registry.list(type="dataset")

