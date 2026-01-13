# Model Lifecycle

This system follows a controlled ML model lifecycle.

## 1. Training
Models are trained using `pipeline/train.py`.  
The pipeline performs:
- Feature preprocessing
- Logistic regression training
- Model evaluation
- Logging to MLflow

## 2. Registration
Each trained model is registered in MLflow under:
LoanDefaultModel

Every training run creates a new version.

## 3. Production Model
One version is marked as the **production (champion)** model.
This is the only model used by the API.

## 4. Challenger Models
When retraining occurs, a new model version becomes a **challenger**.

It is evaluated against the production model using the same test data.

## 5. Promotion
If the challenger outperforms the champion, it is promoted to production.
Otherwise, the existing production model remains active.

This ensures safe and controlled model upgrades.
