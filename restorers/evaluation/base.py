from time import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple

import wandb
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm.auto import tqdm

from ..utils import fetch_wandb_artifact


class BaseEvaluator(ABC):
    def __init__(
        self,
        metrics: List[tf.keras.metrics.Metric],
        model: Optional[tf.keras.Model] = None,
    ) -> None:
        super().__init__()
        self.metrics = metrics
        self.model = model
        self.image_paths = self.populate_image_paths()
        self.wandb_table = self.create_wandb_table() if wandb.run is not None else None

    @abstractmethod
    def preprocess(self, image_path: Image) -> Union[np.ndarray, tf.Tensor]:
        raise NotImplementedError(f"{self.__class__.__name__ }.preprocess")

    @abstractmethod
    def postprocess(self, model_output: np.ndarray) -> Image:
        raise NotImplementedError(f"{self.__class__.__name__ }.postprocess")

    @abstractmethod
    def populate_image_paths(self) -> Dict[str, Tuple[List[str], List[str]]]:
        raise NotImplementedError(f"{self.__class__.__name__ }.populate_image_paths")

    def create_wandb_table(self) -> wandb.Table:
        columns = [
            "Split",
            "Input-Image",
            "Ground-Truth-Image",
            "Enhanced-Image",
            "Inference-Time",
        ]
        metric_alias = [type(metric).__name__ for metric in self.metrics]
        columns = columns + metric_alias
        return wandb.Table(columns=columns)

    def initialize_model_from_wandb_artifact(self, artifact_address: str) -> None:
        model_path = fetch_wandb_artifact(artifact_address, artifact_type="model")
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def evaluate_split(
        self,
        input_image_paths: List[str],
        ground_truth_image_paths: List[str],
        split_name: str,
    ):
        progress_bar = tqdm(
            zip(input_image_paths, ground_truth_image_paths),
            total=len(input_image_paths),
            desc=f"Evaluating {split_name} split",
        )
        for input_image_path, ground_truth_image_path in progress_bar:
            input_image = Image.open(input_image_path)
            ground_truth_image = Image.open(ground_truth_image_path)
            preprocessed_input_image = self.preprocess(input_image)
            preprocessed_ground_truth_image = self.preprocess(ground_truth_image)
            start_time = time()
            model_output = self.model.predict(preprocessed_input_image, verbose=0)
            inference_time = time() - start_time
            metric_results = [
                metric(preprocessed_ground_truth_image, model_output).numpy().item()
                for metric in self.metrics
            ]
            post_processed_image = self.postprocess(model_output)
            if self.wandb_table is not None:
                table_row = [
                    split_name,
                    wandb.Image(input_image),
                    wandb.Image(ground_truth_image),
                    wandb.Image(post_processed_image),
                    inference_time,
                ] + metric_results
                self.wandb_table.add_data(*table_row)

    def evaluate(self):
        for split_name, (
            input_image_paths,
            ground_truth_image_paths,
        ) in self.image_paths.items():
            self.evaluate_split(input_image_paths, ground_truth_image_paths, split_name)
