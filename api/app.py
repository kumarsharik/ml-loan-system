import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from datetime import datetime

# Load PRODUCTION model from registry
MODEL_NAME = "LoanDefaultModel-Production"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI(title="Loan Default Prediction API")


class LoanApplication(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    LoanTerm: int
    DTIRatio: float
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str


@app.post("/predict")
def predict(applicant: LoanApplication):
    data = pd.DataFrame([applicant.dict()])

    prob = model.predict(data)[0]
    decision = "REJECT" if prob > 0.5 else "APPROVE"

    # Log prediction
    log = data.copy()
    log["pd"] = prob
    log["decision"] = decision
    log["timestamp"] = datetime.utcnow().isoformat()

    log.to_csv("monitoring/prediction_log.csv", mode="a", header=False, index=False)

    return {
        "probability_of_default": float(prob),
        "decision": decision
    }