import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_data():
    df = pd.read_csv("data/clean.csv")

    X = df.drop("Default", axis=1)
    y = df["Default"]

    # No shuffle to simulate time-based split
    return train_test_split(X, y, test_size=0.2, shuffle=False)


def build_preprocessor(X):
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numerical = X.select_dtypes(exclude=["object"]).columns.tolist()

    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical),
        ("cat", cat_pipeline, categorical)
    ])

    return preprocessor
