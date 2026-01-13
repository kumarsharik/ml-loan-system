import mlflow
from sklearn.metrics import roc_auc_score
from pipeline.preprocess import get_data

mlflow.set_tracking_uri("http://localhost:5000")

model = mlflow.pyfunc.load_model("models:/LoanDefaultModel-Production/latest")

X_train, X_test, y_train, y_test = get_data()

preds = model.predict(X_test)
auc = roc_auc_score(y_test, preds)

print("Champion AUC:", auc)
