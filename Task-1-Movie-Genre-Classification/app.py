"""Streamlit UI for the trained movie genre classifier."""

from pathlib import Path

import streamlit as st

from src.movie_genre.model import load_model


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models/movie_genre_classifier.joblib"

st.set_page_config(page_title="Movie Genre Classifier", page_icon="🎬")
st.title("🎬 Movie Genre Classification")
st.write("Predict a movie genre from its plot summary using a TF-IDF + LinearSVC model.")

if not MODEL_PATH.is_file():
    st.info("No trained model found. Add the dataset to `data/` and run `python train.py` first.")
else:
    plot = st.text_area("Movie plot summary", height=180, placeholder="Enter the plot summary…")
    if st.button("Predict genre", type="primary"):
        if not plot.strip():
            st.warning("Enter a plot summary first.")
        else:
            try:
                prediction = load_model(MODEL_PATH).predict([plot.strip()])[0]
                st.success(f"Predicted genre: **{prediction}**")
            except Exception as exc:
                st.error(f"Could not load or run the model: {exc}")
