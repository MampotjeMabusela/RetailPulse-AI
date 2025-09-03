import pandas as pd

df_forecast = pd.read_csv("forecast_output.csv")
df_sentiment = pd.read_csv("sentiment_output.csv")
df_pricing = pd.read_csv("pricing_output.csv")
df_segment = pd.read_csv("segmentation_output.csv")

# Merge and export
merged = pd.concat([df_forecast, df_sentiment, df_pricing, df_segment], axis=1)
merged.to_excel("tableau_dashboard_data.xlsx", index=False)
