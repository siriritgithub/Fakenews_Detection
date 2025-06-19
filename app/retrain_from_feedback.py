import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Path to feedback CSV
feedback_path = r"C:\fake-news-detection\app\feedback_log.csv"

# Output paths
model_path = r"C:\fake-news-detection\models\logistic_model.pkl"
vectorizer_path = r"C:\fake-news-detection\models\tfidf_vectorizer.pkl"

# Load feedback data
if not os.path.exists(feedback_path):
    raise FileNotFoundError("Feedback log file not found.")

df = pd.read_csv(feedback_path)

# Check required columns
required_cols = {"headline", "feedback"}
if not required_cols.issubset(df.columns):
    raise ValueError("CSV is missing required columns: headline, feedback")

# Filter valid rows
df = df[df["feedback"].isin(["positive", "negative"])].dropna(subset=["headline"])
df["label"] = df["feedback"].map({"positive": 1, "negative": 0})

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["headline"])
y = df["label"]

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save updated model & vectorizer
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("✅ Retraining complete. Model and vectorizer updated.")
