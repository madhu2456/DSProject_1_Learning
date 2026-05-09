import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
import numpy as np
import joblib
from src.datascience import logger
from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.constants import *
from src.datascience.utils.common import read_yaml, create_directories,save_json
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def _local_mlflow_uri(self) -> str:
        return "sqlite:///mlflow.db"

    def _mlflow_uris(self) -> list[str]:
        if os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv("MLFLOW_TRACKING_PASSWORD"):
            return [self.config.mlflow_uri, self._local_mlflow_uri()]
        return [self._local_mlflow_uri()]

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        for tracking_uri in self._mlflow_uris():
            mlflow.end_run()
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_registry_uri(tracking_uri)
            tracking_url_type_store = urlparse(tracking_uri).scheme

            try:
                with mlflow.start_run():

                    predicted_qualities = model.predict(test_x)

                    (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

                    # Saving metrics as local
                    scores = {"rmse": rmse, "mae": mae, "r2": r2}
                    save_json(path=Path(self.config.metric_file_name), data=scores)

                    mlflow.log_params(self.config.all_params)

                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("r2", r2)
                    mlflow.log_metric("mae", mae)

                    model_info = mlflow.sklearn.log_model(model, "model")

                    # Model registry does not work with file store, and remote registries can deny writes.
                    if tracking_url_type_store != "file":
                        try:
                            mlflow.register_model(model_info.model_uri, "ElasticnetModel")
                        except MlflowException as exc:
                            if "403" in str(exc) or "401" in str(exc):
                                logger.warning(
                                    "Skipping MLflow model registration because the registry rejected the request."
                                )
                            else:
                                raise
                return
            except MlflowException as exc:
                if tracking_uri != self._local_mlflow_uri() and ("403" in str(exc) or "401" in str(exc)):
                    mlflow.end_run()
                    logger.warning(
                        "Remote MLflow tracking was rejected; falling back to a local SQLite store."
                    )
                    continue
                raise
