from prophet import Prophet
import pandas as pd

df = pd.read_csv("sales_data.csv")  # Columns: ds (date), y (sales)
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("forecast_output.csv", index=False)
