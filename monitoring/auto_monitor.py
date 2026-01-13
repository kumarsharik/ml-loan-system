import pandas as pd
import subprocess
import mlflow
from sklearn.metrics import roc_auc_score
from pipeline.preprocess import get_data

COLS = [
    "Age", "Income", "LoanAmount", "CreditScore",
    "MonthsEmployed", "NumCreditLines", "LoanTerm", "DTIRatio",
    "Education", "EmploymentType", "MaritalStatus",
    "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner",
    "pd", "decision", "timestamp"
]

data = pd.read_csv("monitoring/prediction_log.csv", names=COLS)

baseline = data.iloc[:200]
current = data.iloc[-200:]

baseline_pd = baseline["pd"].mean()
current_pd = current["pd"].mean()

drift = abs(current_pd - baseline_pd)

if drift > 0.1:
    print("Drift detected. Training challenger...")
    subprocess.run(["python", "pipeline/train.py"])

    print("Evaluating challenger vs champion")

    X_train, X_test, y_train, y_test = get_data()

    mlflow.set_tracking_uri("http://localhost:5000")

    challenger = mlflow.pyfunc.load_model("models:/LoanDefaultModel/latest")
    champion = mlflow.pyfunc.load_model("models:/LoanDefaultModel-Production/latest")

    auc_challenger = roc_auc_score(y_test, challenger.predict(X_test))
    auc_champion = roc_auc_score(y_test, champion.predict(X_test))

    print("Challenger AUC:", auc_challenger)
    print("Champion AUC:", auc_champion)

    if auc_challenger > auc_champion:
        print("🏆 Challenger wins — promote to production")
        subprocess.run([
            "mlflow", "models", "copy",
            "--src", "models:/LoanDefaultModel/latest",
            "--dst", "LoanDefaultModel-Production"
        ])
    else:
        print("Champion remains")

else:
    print("Model stable — no retraining")
