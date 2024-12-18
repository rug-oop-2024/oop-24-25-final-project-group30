import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import Model
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model import REGRESSION_MODELS, CLASSIFICATION_MODELS, get_model
from autoop.core.ml.metric import METRICS, get_metric

st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.title("Pipeline Modeling")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# Loading Existing Datasets
st.header("Select a Dataset")
datasets = automl.registry.list(type="dataset")  # Retrieve dataset list from the artifact registry
dataset_choice = st.selectbox("Available Datasets", [d['name'] for d in datasets])

if dataset_choice:
    # Load the selected dataset
    selected_dataset = automl.registry.load(name=dataset_choice, type="dataset")
    df = selected_dataset.read()
    st.write("Dataset Preview:")
    st.dataframe(df)

    # Step 2: Detect Features
    st.header("Select Features")
    features = detect_feature_types(selected_dataset)
    feature_names = [f.name for f in features if f.type == 'numerical' or f.type == 'categorical']
    input_features = st.multiselect("Input Features", feature_names)
    target_feature = st.selectbox("Target Feature", feature_names)

    # Retrieve the Feature instances for input and target
    input_feature_instances = [f for f in features if f.name in input_features]
    target_feature_instance = next(f for f in features if f.name == target_feature)

    # Infer Task Type
    task_type = "classification" if df[target_feature].nunique() < 10 else "regression"
    st.write(f"Detected Task Type: {task_type.capitalize()}")

    # Model Selection
    st.header("Select a Model")
    model_choices = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    model_choice = st.selectbox("Choose a Model", model_choices)

    # Get the model instance using get_model
    model = get_model(model_choice)

    # Data split
    st.header("Split the Dataset")
    train_size = st.slider("Training Set Size", 0.1, 0.9, 0.8)
    split_info = f"Train: {int(train_size * 100)}%, Test: {int((1 - train_size) * 100)}%"

    # Select metrics
    st.header("Select Evaluation Metrics")
    if task_type == "classification":
        metric_choices = [
            m for m in METRICS if m in ["accuracy", "precision", "recall", "f1_score"]
        ]
    elif task_type == "regression":
        metric_choices = [
            m for m in METRICS if m in ["mean_squared_error", "mean_absolute_error", "r2_score"]
        ]
    selected_metric_names = st.multiselect("Choose Metrics", metric_choices)

    # Get the metrics using get_metric
    selected_metrics = [get_metric(name) for name in selected_metric_names]

    # Pipeline summary 
    st.header("Pipeline Summary")
    st.write(f"**Dataset**: {dataset_choice}")
    st.write(f"**Input Features**: {', '.join(input_features)}")
    st.write(f"**Target Feature**: {target_feature}")
    st.write(f"**Model**: {model_choice}")
    st.write(f"**Split Ratio**: {split_info}")
    st.write(f"**Metrics**: {', '.join(selected_metric_names)}")

    # Train 
    if st.button("Train Pipeline"):
        # Initialize the pipeline with the selected parameters
        pipeline = Pipeline(
            metrics=selected_metrics,
            dataset=selected_dataset,
            model=model,
            input_features=input_feature_instances,
            target_feature=target_feature_instance,
            split=train_size
        )

        # Execute the pipeline
        results = pipeline.execute()
        
        # Results
        st.success("Training completed!")
        st.write("Training Results:")
        st.json(results) 