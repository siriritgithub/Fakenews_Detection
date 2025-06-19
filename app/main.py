import streamlit as st
import joblib
from get_news import fetch_news
from get_weather import get_weather
from get_cricket import validate_cricket_fact
from get_facts import check_fact
from save_feedback import save_feedback
from performance import show_model_performance
from feedback_dashboard import show_feedback_dashboard
import os

# Load model and vectorizer
model_path = r"C:\fake-news-detection\models\logistic_model.pkl"
vectorizer_path = r"C:\fake-news-detection\models\tfidf_vectorizer.pkl"
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Page config
st.set_page_config(
    page_title="Fake News Detection App",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Title
st.title("📰 Fake News Detection App")
st.subheader("Classify headlines, validate facts, search live news, weather & cricket updates!")

# --- Section 1: Fake News Classifier ---
st.markdown("### 🧠 Fake News Classifier (Custom Headline)")
headline = st.text_input("📩 Enter a headline manually")

if st.button("Predict"):
    if headline.strip() == "":
        st.warning("⚠️ Please enter a headline.")
    else:
        vector_input = vectorizer.transform([headline])
        prediction = model.predict(vector_input)[0]
        confidence = model.predict_proba(vector_input).max() * 100

        if prediction == 1:
            st.success(f"✅ Prediction: 🟢 Real News ({confidence:.2f}% confidence)")
        else:
            st.error(f"❌ Prediction: 🔴 Fake News ({confidence:.2f}% confidence)")

        st.markdown("---")
        st.write("Was this prediction correct?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes"):
                save_feedback(headline, prediction, confidence, "positive")
                st.toast("Thanks for your feedback! 🙏", icon="🟢")
        with col2:
            if st.button("👎 No"):
                save_feedback(headline, prediction, confidence, "negative")
                st.toast("Got it! We'll use this to improve. 💡", icon="🔴")

# --- Section 2: Smart Assistant ---
st.markdown("---")
st.markdown("### 🤖 Smart Assistant — Ask Me Anything")

query = st.text_input("💬 Ask a question or statement (e.g., 'weather in Mumbai', 'CSK won', 'Modi is PM')")

if st.button("🚀 Run Smart Check"):
    if query.strip() == "":
        st.warning("⚠️ Please enter something.")
    else:
        if "weather" in query.lower():
            weather_info = get_weather(query)
            st.info(weather_info)

        elif any(team in query.upper() for team in ["CSK", "KKR", "MI", "RCB", "GT", "LSG", "SRH", "RR", "DC", "PBKS"]):
            result = validate_cricket_fact(query)
            st.info(result)

        elif any(query.lower().startswith(x) for x in ["who", "what", "when", "is", "was", "are"]):
            fact_result = check_fact(query)
            st.info(fact_result)

        else:
            headlines = fetch_news(query)
            if not headlines or "Error" in headlines[0] or "⚠️" in headlines[0]:
                st.warning(headlines[0] if headlines else "No news articles found.")
            else:
                selected = st.selectbox("🗞️ Choose a headline to analyze", headlines)
                if st.button("🧠 Check Selected Headline"):
                    input_vector = vectorizer.transform([selected])
                    prediction = model.predict(input_vector)[0]
                    confidence = model.predict_proba(input_vector).max() * 100

                    if prediction == 1:
                        st.success(f"✅ Prediction: 🟢 Real News ({confidence:.2f}% confidence)")
                    else:
                        st.error(f"❌ Prediction: 🔴 Fake News ({confidence:.2f}% confidence)")

# --- Section 3: Model Performance ---
st.markdown("---")
st.markdown("### 📈 Model Performance Dashboard")

if st.button("📊 Show Model Metrics"):
    show_model_performance()

# --- Section 4: Feedback Stats ---
st.markdown("---")
st.markdown("### 📊 Feedback Insights Dashboard")

if st.button("📈 Show Feedback Stats"):
    show_feedback_dashboard()

# --- Section 5: Download Feedback CSV ---
st.markdown("---")
st.markdown("### 📥 Download Feedback Log")

feedback_path = r"C:\fake-news-detection\app\feedback_log.csv"

try:
    with open(feedback_path, "rb") as f:
        st.download_button(
            label="📁 Download Feedback as CSV",
            data=f,
            file_name="feedback_log.csv",
            mime="text/csv"
        )
except FileNotFoundError:
    st.warning("⚠️ No feedback log found yet.")

# --- Section 6: Retrain Model from Feedback ---
st.markdown("---")
st.markdown("### 🔁 Retrain Model Using Feedback")

if st.button("🧠 Retrain Now"):
    with st.spinner("Retraining model from feedback..."):
        try:
            import subprocess
            subprocess.run(["python", "app/retrain_from_feedback.py"], check=True)
            st.success("✅ Model retrained successfully! Please refresh the app.")
        except Exception as e:
            st.error(f"❌ Retraining failed: {e}")
