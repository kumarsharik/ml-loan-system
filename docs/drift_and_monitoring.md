# Drift and Monitoring

Predictions made by the API are logged to `monitoring/prediction_log.csv`.

The monitoring job:
monitoring/auto_monitor.py

runs periodically and performs:

## 1. Drift Detection
It compares recent predictions with historical baseline predictions.
If the probability of default distribution shifts significantly, drift is flagged.

## 2. Retraining Trigger
When drift is detected:
- A new model is trained
- Registered as a challenger in MLflow

## 3. Model Comparison
The challenger model is compared against the production model using ROC-AUC.

## 4. Promotion
If the challenger performs better, it replaces the production model.

This allows the system to adapt automatically to changing data.
