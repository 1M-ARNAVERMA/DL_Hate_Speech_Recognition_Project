import pandas as pd

df = pd.read_csv("data/labeled_data.csv")

print(df.head())
print(df.shape)
print(df.columns)