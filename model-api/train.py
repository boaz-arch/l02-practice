import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=500).fit(X_train, y_train)
print(f"test accuracy: {model.score(X_test, y_test):.3f}")
joblib.dump(model, "model.joblib")
