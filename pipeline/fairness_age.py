import pandas as pd
import joblib

from preprocess import get_data

# Load trained model
model, preprocessor = joblib.load("model.joblib")

# Load test data
X_train, X_test, y_train, y_test = get_data()

# Get predicted probabilities
X_test_t = preprocessor.transform(X_test)
probs = model.predict_proba(X_test_t)[:, 1]

# Add to dataframe
test = X_test.copy()
test["Default"] = y_test.values
test["PD"] = probs

# Define approval threshold
THRESHOLD = 0.3   # bank risk appetite

test["Decision"] = test["PD"] < THRESHOLD

# Create age buckets
test["AgeGroup"] = pd.cut(
    test["Age"],
    bins=[18, 30, 50, 100],
    labels=["Young", "Mid", "Senior"]
)

# Group stats
summary = test.groupby("AgeGroup").agg(
    approval_rate=("Decision", "mean"),
    default_rate=("Default", "mean"),
    avg_pd=("PD", "mean"),
    count=("Age", "count")
)

print(summary)