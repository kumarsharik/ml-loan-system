import pandas as pd

df = pd.read_csv("data/raw.csv")

# Drop leakage columns
df = df.drop(columns=["LoanID", "InterestRate"])

# Separate features & target
X = df.drop("Default", axis=1)
y = df["Default"]

print("Final feature columns:")
print(X.columns.tolist())

# Save clean dataset
clean = pd.concat([X, y], axis=1)
clean.to_csv("data/clean.csv", index=False)

print("\nSaved clean dataset to data/clean.csv")
