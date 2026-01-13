import joblib
import pandas as pd

model, preprocessor = joblib.load("model.joblib")

# Get feature names after one-hot encoding
feature_names = preprocessor.get_feature_names_out()

coeffs = model.coef_[0]

importance = pd.DataFrame({
    "feature": feature_names,
    "weight": coeffs
})

importance["abs_weight"] = importance["weight"].abs()

# Sort by influence
importance = importance.sort_values("abs_weight", ascending=False)

print(importance.head(20))
