import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from get_news import fetch_news
from get_weather import get_weather
from get_cricket import validate_cricket_fact
from get_facts import check_fact
from performance import show_model_performance  # ✅ Corrected function name
from save_feedback import save_feedback
from feedback_dashboard import display_feedback_dashboard

# Define relative paths for model and feedback file
model_path = os.path.join("models", "logistic_model.pkl")
vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
feedback_path = os.path.join("app", "feedback_log.csv")

# Cache model loading to improve performance
@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model()

# Streamlit app layout
st.set_page_config(page_title="📰 Fake News Detection App", layout="wide")
st.title("📰 Fake News Detection App")

# Define tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 News Check", "🧪 General Fact Check", "🏏 Cricket Fact", "📊 Performance", "📋 Feedback"
])

# 🔍 News Check Tab
with tab1:
    st.subheader("Check if a News Article is Fake or Real")
    user_input = st.text_area("Enter News Text", height=150)

    if st.button("Predict"):
        if user_input.strip():
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]
            if prediction == 1:
                st.success("✅ This is Real News!")
            else:
                st.error("❌ This is Fake News!")

    st.markdown("---")
    st.subheader("Or Search Live News")
    query = st.text_input("Enter a search topic:")
    if st.button("Fetch Live News"):
        if query.strip():
            articles = fetch_news(query)
            if articles:
                for article in articles:
                    st.markdown(f"### {article['title']}")
                    st.write(article.get('description', 'No description available.'))
                    st.markdown(f"[Read More]({article['url']})")
                    st.markdown("---")
            else:
                st.warning("No articles found for this topic.")

# 🧪 General Fact Check Tab
with tab2:
    st.subheader("Wikipedia Fact Check")
    fact = st.text_input("Enter a fact or statement to verify:")
    if st.button("Check Fact"):
        if fact.strip():
            result = check_fact(fact)
            st.write(result)

# 🏏 Cricket Fact Tab
with tab3:
    st.subheader("Validate Cricket Match Result")
    match_fact = st.text_input("Enter a cricket-related statement:")
    if st.button("Validate Cricket Fact"):
        result = validate_cricket_fact(match_fact)
        if result:
            st.success("✅ Fact validated.")
            st.info(result)
        else:
            st.warning("⚠️ Could not validate the fact. Try rephrasing it.")

# 📊 Model Performance Tab
with tab4:
    st.subheader("Model Performance Metrics")
    show_model_performance()  # ✅ Updated function call

# 📋 Feedback Tab
with tab5:
    st.subheader("We value your feedback!")
    feedback = st.text_area("Please share your thoughts or suggestions:")
    if st.button("Submit Feedback"):
        save_feedback(feedback, feedback_path)
        st.success("Thank you for your feedback! ✅")

    st.markdown("### Previous Feedback")
    display_feedback_dashboard(feedback_path)
