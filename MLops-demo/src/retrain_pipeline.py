import os
from pathlib import Path

import mlflow
import mlflow.sklearn

import numpy as np

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from mlflow.models import infer_signature

from src.monitor_drift import check_data_drift


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

EXPERIMENT_NAME = "iris-continuous-retraining"
MODEL_NAME = "IrisRandomForestModel"

MODEL_ARTIFACT_NAME = "iris_random_forest"

MINIMUM_ACCURACY = 0.90

RANDOM_STATE = 42


# --------------------------------------------------
# MLflow CONFIGURATION
# --------------------------------------------------

def configure_mlflow():

    mlflow_home = Path(
        os.getenv("MLFLOW_HOME", "./mlruns")
    ).resolve()

    mlflow_home.mkdir(
        parents=True,
        exist_ok=True
    )

    database_path = mlflow_home / "mlflow.db"

    tracking_uri = (
        f"sqlite:///{database_path.as_posix()}"
    )

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_registry_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)


# --------------------------------------------------
# TEST DATA AND MODEL TRAINING
# --------------------------------------------------

def train_and_register_model():

    print("\nStarting Iris model retraining...")

    configure_mlflow()

    # Load Iris dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Initialize Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=RANDOM_STATE
    )

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nCandidate Model Accuracy: {accuracy:.4f}"
    )

    # Generate MLflow signature
    signature = infer_signature(
        X_train,
        model.predict(X_train)
    )

    # --------------------------------------------------
    # LOG MODEL AND REGISTER MODEL
    # --------------------------------------------------

    with mlflow.start_run(
        run_name="iris-automated-retraining"
    ) as run:

        # Log parameters
        mlflow.log_params({

            "model_type": "RandomForestClassifier",

            "n_estimators": 200,

            "max_depth": 5,

            "random_state": RANDOM_STATE,

            "test_size": 0.20

        })

        # Log accuracy
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        # Log model
        mlflow.sklearn.log_model(

            sk_model=model,

            name=MODEL_ARTIFACT_NAME,

            signature=signature,

            input_example=X_test[:5],

            registered_model_name=MODEL_NAME,

            # FIX FOR YOUR MLflow UNTRUSTED TYPE ERROR
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]

        )

        print(
            "\nMLflow Run ID:",
            run.info.run_id
        )

        print(
            "\nModel registered successfully:",
            MODEL_NAME
        )

    return accuracy


# --------------------------------------------------
# AUTOMATED RETRAINING PIPELINE
# --------------------------------------------------

def run_pipeline():

    print("\n===================================")
    print(" IRIS AUTOMATED RETRAINING PIPELINE")
    print("===================================")

    configure_mlflow()

    # Check production data for drift
    print(
        "\nChecking production data for drift..."
    )

    drift_status = check_data_drift()

    if drift_status:

        print(
            "\nData drift confirmed!"
        )

        print(
            "\nInitiating retraining process..."
        )

        accuracy = train_and_register_model()

        print(
            "\nCandidate Model Accuracy:",
            accuracy
        )

        if accuracy >= MINIMUM_ACCURACY:

            print(
                "\nCandidate model meets accuracy threshold."
            )

            print(
                "Model successfully registered in MLflow."
            )

        else:

            print(
                "\nCandidate model did not meet accuracy threshold."
            )

    else:

        print(
            "\nNo significant data drift detected."
        )

        print(
            "Retraining is not required."
        )

    print(
        "\n==================================="
    )

    print(
        " PIPELINE EXECUTION COMPLETED"
    )

    print(
        "===================================")


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    run_pipeline()
