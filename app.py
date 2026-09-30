
import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer

model = joblib.load("sentiment_embedding_lr.pkl")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("Sentiment Analyzer")
st.write("Enter a sentence or review to predict its sentiment.")

review = st.text_area(
    "Enter your text:",
    placeholder="Example: The movie was amazing and I really enjoyed it!"
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter some text.")

    else:
        embedding = embedding_model.encode([review])
        prediction = model.predict(embedding)[0]

        if prediction == 1:
            st.success("Positive Sentiment")
        else:
            st.error("Negative Sentiment")
