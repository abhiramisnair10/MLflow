import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

try:
    from .monitor_drift import check_data_drift
except ImportError:
    from monitor_drift import check_data_drift


# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-continuous-retraining")

MODEL_NAME = "IrisRandomForestModel"
ACCURACY_THRESHOLD = 0.90
client = MlflowClient()


def train_and_register_model():

    print("Initiating retraining process...")

    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42
    )

    # Check existing promoted model
    baseline_accuracy = None

    try:
        baseline = mlflow.sklearn.load_model(
            f"models:/{MODEL_NAME}@champion"
        )

        baseline_predictions = baseline.predict(X_test)
        baseline_accuracy = accuracy_score(
            y_test, baseline_predictions
        )

        print(f"Current champion accuracy: {baseline_accuracy:.4f}")

    except Exception:
        print("No existing champion model. First promotion is possible.")

    with mlflow.start_run(run_name="automated_retrain_run"):

        n_estimators = 100
        max_depth = 5

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("trigger", "data_drift")
        mlflow.log_metric("accuracy", accuracy)

        print(f"Candidate Model Accuracy: {accuracy:.4f}")

        # Quality gate
        if accuracy < ACCURACY_THRESHOLD:
            print("Quality threshold failed. Promotion skipped.")
            return

        # Compare candidate with current champion
        if (
            baseline_accuracy is not None
            and accuracy <= baseline_accuracy
        ):
            print("Candidate did not outperform champion. Promotion skipped.")
            return

        # Register candidate
        mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name=MODEL_NAME,
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
        # Find the newest registered version
        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )

        newest_version = max(
            versions,
            key=lambda version: int(version.version)
        )

        # Promote candidate using champion alias
        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            newest_version.version
        )

        print(
            f"Model registered: version {newest_version.version}"
        )
        print("Model promoted to champion successfully!")


def run_pipeline():

    print("Checking production data for drift...")

    drift_detected = check_data_drift()

    if drift_detected:

        print(
            "Data drift confirmed! Triggering retraining pipeline..."
        )

        train_and_register_model()

    else:

        print(
            "No significant drift detected. "
            "Model operating within normal bounds."
        )


if __name__ == "__main__":
    run_pipeline()