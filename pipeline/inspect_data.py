import pandas as pd

df = pd.read_csv("data/raw.csv")

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Columns ---")
print(df.columns.tolist())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isna().sum())

print("\n--- Possible target columns ---")
for col in df.columns:
    if "default" in col.lower() or "target" in col.lower() or "loan" in col.lower():
        print(f"{col}:")
        print(df[col].value_counts())
