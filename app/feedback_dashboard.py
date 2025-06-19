import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def show_feedback_dashboard():
    st.title("📊 Feedback Insights Dashboard")

    feedback_file = r"C:\fake-news-detection\app\feedback_log.csv"

    try:
        df = pd.read_csv(feedback_file)

        # Drop rows with missing feedback
        df = df.dropna(subset=["feedback"])

        # Ensure feedback values are lowercase for consistency
        df["feedback"] = df["feedback"].str.lower().str.strip()

        # Count feedback types
        pos = int((df["feedback"] == "positive").sum())
        neg = int((df["feedback"] == "negative").sum())

        if pos + neg == 0:
            st.warning("⚠️ No valid feedback data found yet. Try giving some feedback first.")
            return

        # Pie chart
        fig, ax1 = plt.subplots()
        ax1.pie([pos, neg], labels=["Positive", "Negative"], autopct="%1.1f%%", startangle=90, colors=["green", "red"])
        ax1.axis("equal")
        st.pyplot(fig)

        st.write("### Raw Feedback Data")
        st.dataframe(df)

    except FileNotFoundError:
        st.warning("⚠️ Feedback log file not found.")
    except Exception as e:
        st.error(f"⚠️ Error loading feedback dashboard: {e}")