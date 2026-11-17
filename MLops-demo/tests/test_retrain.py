import mlflow

from src.monitor_drift import check_data_drift
from src.retrain_pipeline import (
    train_and_register_model,
    run_pipeline,
)


def test_data_drift_detection():
    drift_status = check_data_drift()
    assert isinstance(drift_status, bool)


def test_train_and_register_model_execution():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    train_and_register_model()

    experiment = mlflow.get_experiment_by_name(
        "iris-continuous-retraining"
    )

    assert experiment is not None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    assert len(runs) > 0
    assert runs.iloc[0]["metrics.accuracy"] >= 0.90


def test_full_retrain_pipeline_trigger(capsys):
    run_pipeline()

    captured = capsys.readouterr()

    assert "Checking production data for drift..." in captured.out
