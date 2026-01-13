import pandas as pd

cols = [
    "Age", "Income", "LoanAmount", "CreditScore",
    "MonthsEmployed", "NumCreditLines", "LoanTerm", "DTIRatio",
    "Education", "EmploymentType", "MaritalStatus",
    "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner",
    "pd", "decision", "timestamp"
]

data = pd.read_csv("monitoring/prediction_log.csv", names=cols)

baseline = data.iloc[:200]
current = data.iloc[-200:]

baseline_pd = baseline["pd"].mean()
current_pd = current["pd"].mean()

print("Baseline PD:", round(baseline_pd, 3))
print("Current PD:", round(current_pd, 3))

drift = abs(current_pd - baseline_pd)

if drift > 0.1:
    print("⚠️  Prediction drift detected!")
else:
    print("Model stable")
