import pandas as pd

df = pd.read_csv("data/clean.csv")

categorical = df.select_dtypes(include=["object"]).columns.tolist()
numerical = df.select_dtypes(exclude=["object"]).drop("Default", axis=1).columns.tolist()

print("Categorical:", categorical)
print("Numerical:", numerical)