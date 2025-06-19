# app/performance.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

def show_model_performance():
    # Load data
    df = pd.read_csv(r"C:\fake-news-detection\data\politifact_cleaned.csv")

    # Prepare data
    df = df.dropna(subset=["clean_title"])
    vectorizer = joblib.load(r"C:\fake-news-detection\models\tfidf_vectorizer.pkl")
    model = joblib.load(r"C:\fake-news-detection\models\logistic_model.pkl")

    X = vectorizer.transform(df["clean_title"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    # Show classification report
    st.markdown("### 📋 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(2))

    # Confusion matrix
    st.markdown("### 🔁 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'], ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Bar chart of metrics
    st.markdown("### 📊 Precision, Recall, F1-Score")
    metrics = report_df.loc[["0", "1"], ["precision", "recall", "f1-score"]]
    st.bar_chart(metrics)
