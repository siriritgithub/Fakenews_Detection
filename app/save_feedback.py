import pandas as pd
import os

def save_feedback(headline, prediction, confidence, feedback):
    feedback_file = r"C:\fake-news-detection\app\feedback_log.csv"

    data = {
        "headline": [headline],
        "prediction": [prediction],
        "confidence": [confidence],
        "feedback": [feedback]
    }

    df = pd.DataFrame(data)

    # Append to file if exists, else create with header
    if os.path.exists(feedback_file):
        df.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        df.to_csv(feedback_file, mode='w', header=True, index=False)
