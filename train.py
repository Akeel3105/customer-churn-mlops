import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# 1. Load dataset
data = load_breast_cancer()

X = data.data
y = data.target


# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 3. Random Forest
model = RandomForestClassifier(
    random_state=42
)


# 4. Hyperparameter combinations
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15]
}


# 5. GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


# 6. MLflow tracking
with mlflow.start_run(run_name="Random Forest - GridSearchCV"):

    # Hyperparameter tuning using only training data
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Final prediction on untouched test data
    y_pred = best_model.predict(X_test)

    # Final metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)


    # Log best parameters
    mlflow.log_param(
        "best_n_estimators",
        grid_search.best_params_["n_estimators"]
    )

    mlflow.log_param(
        "best_max_depth",
        grid_search.best_params_["max_depth"]
    )


    # Log CV performance
    mlflow.log_metric(
        "best_cv_f1",
        grid_search.best_score_
    )


    # Log final test performance
    mlflow.log_metric(
        "test_accuracy",
        test_accuracy
    )

    mlflow.log_metric(
        "test_f1",
        test_f1
    )


    # Save and register best model
    mlflow.sklearn.log_model(
        best_model,
        "model",
        registered_model_name="RandomForestBreastCancer"
    )


    # Print results
    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print(f"\nBest CV F1: {grid_search.best_score_:.4f}")

    print(f"Final Test Accuracy: {test_accuracy:.4f}")

    print(f"Final Test F1: {test_f1:.4f}")