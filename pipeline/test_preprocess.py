from preprocess import get_data, build_preprocessor

X_train, X_test, y_train, y_test = get_data()
preprocessor = build_preprocessor(X_train)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

print("Train shape:", X_train_transformed.shape)
print("Test shape:", X_test_transformed.shape)
