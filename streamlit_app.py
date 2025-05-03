# streamlit_app.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.preprocessing import TextCleaner  # adjust if your class is elsewhere
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("models/best_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🎭 IMDB Review Sentiment Analyzer")
st.markdown("Enter a movie review below, and see if it's **Positive** or **Negative**.")

text_input = st.text_area("Your Review", height=200)

if st.button("Analyze") and text_input:
    # Preprocess and vectorize
    cleaner = TextCleaner()
    cleaned = cleaner.transform(pd.Series([text_input]))
    vec = tfidf.transform(cleaned).toarray()

    # Predict
    pred = model.predict(vec)[0]
    label = "🟢 Positive" if pred == 1 else "🔴 Negative"
    st.markdown(f"### Prediction: {label}")

    # SHAP Explanation
    with st.spinner("Generating explanation..."):
        explainer = shap.Explainer(model, vec)
        shap_values = explainer(vec)
        
        st.markdown("#### 🔍 Model Explanation:")
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
else:
    st.markdown("_Awaiting input..._")
