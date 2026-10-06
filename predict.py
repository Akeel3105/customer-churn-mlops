import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer


# 1. Load model from MLflow Model Registry
model = mlflow.sklearn.load_model(
    "models:/RandomForestBreastCancer/1"
)


# 2. Load dataset
data = load_breast_cancer()

X = data.data
y = data.target


# 3. Make prediction
prediction = model.predict(X[:5])


# 4. Show results
print("Predictions:", prediction)
print("Actual values:", y[:5])