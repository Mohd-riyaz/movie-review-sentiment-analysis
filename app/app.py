import streamlit as st
import joblib
import re
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review and I'll predict whether the sentiment is Positive or Negative.")
review = st.text_area(
    "Enter your movie review:",
    height=200,
    placeholder="Example: This movie was amazing! I loved every minute of it."
)
predict_button = st.button("🔍 Predict Sentiment")
