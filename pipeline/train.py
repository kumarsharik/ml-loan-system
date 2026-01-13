import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from preprocess import get_data, build_preprocessor


def train():
    # Load data
    X_train, X_test, y_train, y_test = get_data()

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(X_train)

    # Define model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=0.5
    )

    # Build full ML pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Start MLflow run
    with mlflow.start_run():

        # Train full pipeline
        pipeline.fit(X_train, y_train)

        # Predict probabilities
        probs = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)

        print("ROC-AUC:", auc)

        # Log parameters
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("C", 0.5)

        # Log metrics
        mlflow.log_metric("roc_auc", auc)

        # Log FULL pipeline (this is the key fix)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="LoanDefaultModel"
        )

        print("Model + Preprocessor logged and registered in MLflow")


if __name__ == "__main__":
    train()
