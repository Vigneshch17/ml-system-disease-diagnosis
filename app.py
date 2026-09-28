"""Streamlit demo for the saved four-class model."""

from pathlib import Path

import streamlit as st
from PIL import Image

from src.data import image_to_features
from src.predict import load_bundle, predict_features


st.set_page_config(page_title="Chest X-Ray detection", page_icon="🩻")
st.title("Chest X-Ray detection")
st.warning(
    "Educational research demo only. This model is not clinically validated and must not "
    "be used to make medical decisions. Consult a qualified healthcare professional."
)
st.write("Upload an image to obtain a model prediction for Normal, Tuberculosis, Pneumonia, or COVID.")
model_path = Path("artifacts/model.joblib")
if not model_path.is_file():
    st.info("No trained model found. Run `python -m src.train --data lbddataset.csv` first.")
    st.stop()

uploaded = st.file_uploader("Chest X-ray image", type=["png", "jpg", "jpeg", "bmp"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)
    st.caption("Assumption: grayscale 28×28 resizing and row-major flattening match training features.")
    try:
        bundle = load_bundle(model_path)
        prediction, probabilities = predict_features(image_to_features(image), bundle)[0]
        st.subheader(f"Model output: {prediction}")
        if probabilities:
            st.write("Class scores (model probability estimates):")
            st.bar_chart(probabilities)
        st.caption(f"Classifier: {bundle['model_name']}")
    except Exception as error:
        st.error(f"Could not run prediction: {error}")
