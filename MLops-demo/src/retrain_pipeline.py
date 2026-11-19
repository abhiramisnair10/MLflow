import os
from pathlib import Path

import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "IrisRandomForestModel"
EXPERIMENT_NAME = "iris-continuous-retraining"

ACCURACY_THRESHOLD = 0.90


# --------------------------------------------------
# MLflow CONFIGURATION
# --------------------------------------------------

def configure_mlflow():
    """
    Configure MLflow with cross-platform local storage.

    MLFLOW_HOME can be overridden by tests or CI.
    """

    mlflow_dir = Path(
        os.getenv("MLFLOW_HOME", str(BASE_DIR / "mlflow_data"))
    ).resolve()

    mlflow_dir.mkdir(parents=True, exist_ok=True)

    database_path = mlflow_dir / "mlflow.db"
    artifact_dir = mlflow_dir / "artifacts"

    artifact_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{database_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        try:
            client.create_experiment(
                name=EXPERIMENT_NAME,
                artifact_location=artifact_dir.as_uri()
            )
        except Exception:
            # Handle an experiment created concurrently.
            experiment = client.get_experiment_by_name(
                EXPERIMENT_NAME
            )

            if experiment is None:
                raise

    else:
        artifact_location = experiment.artifact_location

        # Detect an old Windows artifact path in a Linux environment.
        if artifact_location.startswith("/C:/") or (
            len(artifact_location) > 2
            and artifact_location[1:3] == ":/"
        ):
            raise RuntimeError(
                "MLflow experiment contains an invalid Windows "
                "artifact location. Remove the old mlflow.db "
                "and create a fresh MLflow database."
            )

    mlflow.set_experiment(EXPERIMENT_NAME)

    return client


# --------------------------------------------------
# MODEL TRAINING AND REGISTRATION
# --------------------------------------------------

def train_and_register_model():

    print("Initiating retraining process...")

    client = configure_mlflow()

    # Load Iris dataset
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42
    )

    # --------------------------------------------------
    # CHECK EXISTING CHAMPION MODEL
    # --------------------------------------------------

    baseline_accuracy = None

    try:
        baseline = mlflow.sklearn.load_model(
            f"models:/{MODEL_NAME}@champion"
        )

        baseline_predictions = baseline.predict(X_test)

        baseline_accuracy = accuracy_score(
            y_test,
            baseline_predictions
        )

        print(
            f"Current champion accuracy: "
            f"{baseline_accuracy:.4f}"
        )

    except Exception:
        print(
            "No existing champion model. "
            "First promotion is possible."
        )

    # --------------------------------------------------
    # TRAIN CANDIDATE MODEL
    # --------------------------------------------------

    n_estimators = 100
    max_depth = 5

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    with mlflow.start_run(
        run_name="automated_retrain_run"
    ):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            f"Candidate Model Accuracy: {accuracy:.4f}"
        )

        # Log parameters
        mlflow.log_param(
            "n_estimators",
            n_estimators
        )

        mlflow.log_param(
            "max_depth",
            max_depth
        )

        mlflow.log_param(
            "model_type",
            "RandomForestClassifier"
        )

        # Log evaluation metric
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        # Log and register model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

    # --------------------------------------------------
    # CHAMPION MODEL PROMOTION
    # --------------------------------------------------

    if (
        accuracy >= ACCURACY_THRESHOLD
        and (
            baseline_accuracy is None
            or accuracy > baseline_accuracy
        )
    ):

        versions = client.get_latest_versions(
            MODEL_NAME,
            stages=["None"]
        )

        if not versions:
            raise RuntimeError(
                "Registered model version not found."
            )

        candidate_version = versions[0].version

        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            candidate_version
        )

        print(
            f"Model version {candidate_version} "
            "promoted to champion."
        )

    else:
        print(
            "Candidate model did not outperform "
            "the champion or meet the accuracy threshold. "
            "Champion remains unchanged."
        )

    return accuracy


# --------------------------------------------------
# COMPLETE AUTOMATED RETRAINING PIPELINE
# --------------------------------------------------

def run_pipeline():

    # Import drift detection
    try:
        from .monitor_drift import check_data_drift
    except ImportError:
        from monitor_drift import check_data_drift

    print("Checking production data for drift...")

    drift_detected = check_data_drift()

    if drift_detected:

        print(
            "Data drift confirmed! "
            "Triggering retraining pipeline..."
        )

        accuracy = train_and_register_model()

        return accuracy

    else:

        print(
            "No significant data drift detected. "
            "Retraining skipped."
        )

        return None


# --------------------------------------------------
# SCRIPT ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    run_pipeline()
