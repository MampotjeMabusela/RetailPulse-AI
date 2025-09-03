import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("customer_data.csv")  # Features: frequency, recency, monetary
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=4)
df["segment"] = kmeans.fit_predict(X_scaled)
df.to_csv("segmentation_output.csv", index=False)
