import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("pricing_data.csv")  # Features: demand, cost, competitor_price
X = df[["demand", "cost", "competitor_price"]]
y = df["optimal_price"]

model = LinearRegression()
model.fit(X, y)

df["recommended_price"] = model.predict(X)
df.to_csv("pricing_output.csv", index=False)
