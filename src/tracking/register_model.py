"""Register the trained main model in MLflow if gating criteria are met."""

import json
from pathlib import Path
import yaml
import mlflow

from src.tracking.mlflow_utils import configure_mlflow, assign_stage_or_alias


def main() -> None:
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    mlflow_cfg = params["mlflow"]
    if not mlflow_cfg.get("register_model", False):
        print("Model registration is disabled in params.yaml (mlflow.register_model=false). Skipping.")
        return

    metrics_path = Path("reports/metrics/main_metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    macro_f1 = metrics.get("macro_f1", 0.0)
    min_f1 = mlflow_cfg.get("min_f1_to_register", 0.70)
    if macro_f1 < min_f1:
        print(f"Macro F1 ({macro_f1:.4f}) is below threshold ({min_f1:.4f}). Skipping registration.")
        return

    client = configure_mlflow(mlflow_cfg["experiment_name"])
    model_name = mlflow_cfg["registered_model_name"]
    target_stage = mlflow_cfg.get("target_stage", "Staging")

    with mlflow.start_run(run_name="register_main_model") as run:
        mlflow.log_artifact("models/resnet18_main.pth", artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
        assign_stage_or_alias(client, model_name, mv.version, stage=target_stage)
        print(f"Registered {model_name} v{mv.version} -> {target_stage}")


if __name__ == "__main__":
    main()
