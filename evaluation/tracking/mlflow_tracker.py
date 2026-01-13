from __future__ import annotations

from typing import Any, Dict, Optional


class MLflowTracker:
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            import mlflow  # type: ignore
        except Exception as exc:
            raise RuntimeError("mlflow is required but not installed") from exc

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        self._run_name = run_name
        self._tags = tags
        self._active_run = None

    def set_experiment(self, name: str) -> None:
        self._mlflow.set_experiment(name)

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, Any]] = None):
        if self._active_run is None:
            self._active_run = self._mlflow.start_run(
                run_name=run_name or self._run_name,
                tags=tags or self._tags,
            )
        return self._active_run

    def end_run(self, status: str = "FINISHED") -> None:
        if self._mlflow.active_run():
            self._mlflow.end_run(status=status)
        self._active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        if params:
            self._mlflow.log_params(params)

    def log_param(self, key: str, value: Any) -> None:
        self._mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if metrics:
            self._mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        self._mlflow.log_metric(key, value, step=step)

    def set_tags(self, tags: Dict[str, Any]) -> None:
        if tags:
            self._mlflow.set_tags(tags)

    def set_tag(self, key: str, value: Any) -> None:
        self._mlflow.set_tag(key, value)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        self._mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        self._mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

    def __enter__(self) -> "MLflowTracker":
        self.start_run()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status=status)
