from transformers import pipeline
import pandas as pd

df = pd.read_csv("customer_feedback.csv")  # Column: text
sentiment_pipeline = pipeline("sentiment-analysis")

df["sentiment"] = df["text"].apply(lambda x: sentiment_pipeline(x)[0]["label"])
df.to_csv("sentiment_output.csv", index=False)
