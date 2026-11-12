"""MLflow and DagsHub tracking configuration utilities."""

import os
from pathlib import Path
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient


def configure_mlflow(experiment_name: str) -> MlflowClient:
    """Load credentials from .env, configure MLflow tracking URI, and set experiment."""
    load_dotenv(Path(".env"))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return MlflowClient()


def assign_stage_or_alias(
    client: MlflowClient,
    model_name: str,
    version: str,
    stage: str = "Staging",
) -> None:
    """Assign a model version stage with fallback to modern MLflow aliases and tags."""
    alias = stage.lower()
    try:
        client.set_registered_model_alias(name=model_name, alias=alias, version=version)
    except Exception:
        pass

    try:
        client.set_model_version_tag(
            name=model_name, version=version, key="stage", value=stage
        )
    except Exception:
        pass

    try:
        client.transition_model_version_stage(
            name=model_name, version=version, stage=stage
        )
    except Exception:
        # Newer MLflow versions deprecate or disable stage transitions
        pass
