import mlflow

from src.monitor_drift import check_data_drift

from src.retrain_pipeline import (
    train_and_register_model,
    run_pipeline,
)


# --------------------------------------------------
# TEST 1: DATA DRIFT DETECTION
# --------------------------------------------------

def test_data_drift_detection():

    drift_status = check_data_drift()

    assert isinstance(drift_status, bool)


# --------------------------------------------------
# TEST 2: MODEL TRAINING AND REGISTRATION
# --------------------------------------------------

def test_train_and_register_model_execution(
    tmp_path,
    monkeypatch
):

    # Use isolated MLflow storage for this test.
    mlflow_home = tmp_path / "mlflow_test"

    monkeypatch.setenv(
        "MLFLOW_HOME",
        str(mlflow_home)
    )

    # Execute retraining
    accuracy = train_and_register_model()

    assert accuracy >= 0.90

    # Verify MLflow experiment
    experiment = mlflow.get_experiment_by_name(
        "iris-continuous-retraining"
    )

    assert experiment is not None

    # Verify MLflow runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    assert len(runs) > 0

    # Verify recorded accuracy
    assert runs.iloc[0]["metrics.accuracy"] >= 0.90

    # Verify registered model
    client = mlflow.tracking.MlflowClient()

    registered_model = client.get_registered_model(
        "IrisRandomForestModel"
    )

    assert registered_model is not None


# --------------------------------------------------
# TEST 3: COMPLETE AUTOMATED RETRAINING PIPELINE
# --------------------------------------------------

def test_full_retrain_pipeline_trigger(
    capsys,
    tmp_path,
    monkeypatch
):

    # Use separate MLflow storage for pipeline test.
    mlflow_home = tmp_path / "pipeline_mlflow"

    monkeypatch.setenv(
        "MLFLOW_HOME",
        str(mlflow_home)
    )

    # Execute complete pipeline
    run_pipeline()

    # Capture console output
    captured = capsys.readouterr()

    # Verify drift monitoring executed
    assert (
        "Checking production data for drift..."
        in captured.out
    )

    # If drift is detected, retraining must execute.
    if "Data drift confirmed!" in captured.out:

        assert (
            "Initiating retraining process..."
            in captured.out
        )

        assert (
            "Candidate Model Accuracy:"
            in captured.out
        )

    else:

        assert (
            "No significant data drift detected."
            in captured.out
        )
