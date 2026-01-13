# Loan Default ML System

This repository implements an end-to-end machine learning system for predicting loan default risk.  
It covers the full production lifecycle: feature engineering, model training, model registry, API serving, monitoring, drift detection, and automated retraining.

The goal is to demonstrate how a machine learning model is managed and operated in a real production setting.

---

## Problem

Given loan applicant data (income, credit score, employment, etc.), predict the probability that the applicant will default on their loan.

The prediction is used to drive decisions:
- Approve
- Manual review
- Reject

---

## System Architecture

Client  
→ FastAPI (/predict)  
→ MLflow Model Registry (Production model)  
→ Prediction Log  
→ Drift Monitor  
→ Retraining (if needed)

---

## Project Structure

ml-loan-system  
├── pipeline  
├── api  
├── monitoring  
├── data  
├── mlruns  
└── requirements.txt  

---

## Training & Model Registry

The training pipeline:
- Preprocesses data  
- Trains a logistic regression model  
- Logs metrics and artifacts to MLflow  
- Registers the model

Multiple model versions are stored in MLflow.

---

## Inference API

Start API:

export MLFLOW_TRACKING_URI=http://localhost:5000

uvicorn api.app:app --reload


Call API:

POST /predict

Returns:
- Probability of default
- Decision

---

## Monitoring & Retraining

Predictions are logged.

The monitoring job:
- Detects drift
- Trains a challenger
- Compares to champion
- Promotes if better

---

## How to Run

Install:

pip install -r requirements.txt

Train:

python pipeline/train.py

Start MLflow:

mlflow ui
mlflow ui

Start API:

uvicorn api.app:app --reload

Run monitoring:

python monitoring/auto_monitor.py

---

## Technologies

- Python  
- scikit-learn  
- MLflow  
- FastAPI  
- Pandas  
