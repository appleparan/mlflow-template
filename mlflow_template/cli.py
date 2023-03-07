"""CLI for mlops_template."""
import datetime
import logging
from pathlib import Path

import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger("mlflow")


def _explore_recursive(parent_name: str, element: dict) -> None:
    """Log params from a dict or a list.

    Args:
        parent_name (str): parent name
        element (dict): element to log
    """
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{k}", v) # type: ignore
            else:
                mlflow.log_param(f"{parent_name}.{k}", v) # type: ignore
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f"{parent_name}.{i}", v)


def log_params_from_omegaconf_dict(params: dict) -> None:
    """log params from a dict

    Args:
        params (dict): params to log
    """
    for param_name, element in params.items():
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    _explore_recursive(f"{param_name}.{k}", v) # type: ignore
                else:
                    mlflow.log_param(f"{param_name}.{k}", v) # type: ignore
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                mlflow.log_param(f"{param_name}.{i}", v)


def search_run(
    client: mlflow.tracking.MlflowClient,
    experiment_id: str,
    is_resume: bool = False,
    run_name: str = "default_run_name",
) -> mlflow.entities.Run:
    """Search for a run with the same name and return it if it exists.

    Args:
        client (mlflow.tracking.MlflowClient): mlflow client
        experiment_id (str): experiment id
        is_resume (bool): Whether to resume the run. Defaults to False.
        run_name (str): run name. Defaults to "default_run_name".

    Raises:
        ValueError: if there are more than one runs with the same name

    Returns:
        mlflow.entities.Run: run
    """
    if is_resume is False:
        return client.create_run(experiment_id=experiment_id, run_name=run_name)

    existed_runs = client.search_runs(
        experiment_id=[experiment_id], filter_string=f"tags.mlflow.runName='{run_name}'"
    )

    if existed_runs is None or len(existed_runs) == 0:
        return client.create_run(experiment_id=experiment_id, run_name=run_name)
    elif len(existed_runs) == 1:
        return client.get_run(existed_runs.loc[0, "run_id"])
    else:
        raise ValueError(
            "There are more than one runs with the same name. Please check your run name."
        )


@hydra.main(config_path="../configs.sample", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run the main function.

    Args:
        cfg (DictConfig): Config

    Raises:
        ValueError: if there are more than one runs with the same name
    """
    logger.info("Arguments: %s", OmegaConf.to_yaml(cfg))
    logger.info("Current directory: %s", Path.cwd())
    logger.info("Current time: %s", datetime.datetime.now())
    logger.info("Hydra Output Directory: %s", hydra.utils.get_original_cwd())

    if cfg.name.lower() == "model_train":
        tracking_dir = Path(hydra.utils.get_original_cwd()) / "mlruns"
        tracking_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_dir.as_uri())

        client = mlflow.tracking.MlflowClient()

        # if an experiment exists, load it, otherwise create a new one
        try:
            experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
            experiment_id = experiment.experiment_id
        except AttributeError:
            experiment_id = client.create_experiment(
                cfg.mlflow.experiment_name, tags=cfg.mlflow.tags
            )
            experiment = mlflow.get_experiment(experiment_id)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

        # if a run exists, load it, otherwise create a new one
        default_run_name: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = default_run_name if cfg.mlflow.run_name is None else str(cfg.mlflow.run_name)
        run = search_run(client, experiment_id, is_resume=cfg.mlflow.resume, run_name=run_name)

        with mlflow.start_run(run_id=run.info.run_id):
            output_dir = Path(hydra.utils.get_original_cwd()) / Path(HydraConfig.get().run.dir)

            logger.info("Experiment name: %s", experiment.name)
            logger.info("Experiment id: %s", experiment_id)
            logger.info("Artifact URI: %s", experiment.artifact_location)
            logger.info("Tags: %s", experiment.tags)
            logger.info("Lifecycle stages: %s", experiment.lifecycle_stage)
            logger.info("Run name: %s", run.info.run_name)
            logger.info("Run id: %s", run.info.run_id)
            logger.info("Hydra Output Directory: %s", output_dir)
            log_params_from_omegaconf_dict(cfg)

            # Log Hydra config as artifact
            if cfg.mlflow.resume is False:
                mlflow.log_artifact(output_dir / ".hydra")

            # Run model

            # Terminate client
            client.set_terminated(run.info.run_id)
    else:
        raise ValueError("Invalid name")


if __name__ == "__main__":
    # pylint: disable = no-value for parameter
    main()
