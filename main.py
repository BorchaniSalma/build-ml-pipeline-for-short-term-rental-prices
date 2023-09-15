"""
This is our main file which run all of our component

Date:09/15/2023
Author: Salma Borchani

"""



import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

       
        if "basic_cleaning" in active_steps:
            # running the basic_cleaning component
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                # version = "main"
                parameters={
                    "input_artifact": config["parameters"]["basic_cleaning"]["input_artifact"],
                    "output_artifact": config["parameters"]["basic_cleaning"]["output_artifact"],
                    "artifact_type": config["parameters"]["basic_cleaning"]["artifact_type"],
                    "artifact_description":
                      config["parameters"]["basic_cleaning"]["artifact_description"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

        if "data_check" in active_steps:
            # running the data_check component
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                # version = "main"
                parameters={
                    "csv": config["parameters"]["data_check"]["csv"],
                    "ref": config["parameters"]["data_check"]["ref"],
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                }
            )

        if "data_split" in active_steps:
            # running the data_split component
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "train_val_test_split"),
                "main",
                # version = "main"
                parameters={
                    "input": config["parameters"]["data_split"]["input"],
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                }
            )
        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            ##################
            # Implement here #
            ##################

            pass

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################

            pass


if __name__ == "__main__":
    go()
