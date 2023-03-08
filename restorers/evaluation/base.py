from time import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple

import wandb
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm.auto import tqdm

from ..utils import fetch_wandb_artifact, count_params, calculate_gflops


class BaseEvaluator(ABC):
    def __init__(
        self,
        metrics: List[tf.keras.metrics.Metric],
        model: Optional[tf.keras.Model] = None,
        input_size: Optional[int] = None,
        resize_target: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.metrics = metrics
        self.model = model
        self.input_size = input_size
        self.resize_target = resize_target
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
        self.model_path = fetch_wandb_artifact(artifact_address, artifact_type="model")
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

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
        total_metric_values = [0.0] * len(self.metrics)
        total_inference_time = 0
        for input_image_path, ground_truth_image_path in progress_bar:
            input_image = Image.open(input_image_path)
            ground_truth_image = Image.open(ground_truth_image_path)
            if self.resize_target is not None:
                input_image = input_image.resize(self.resize_target[::-1])
                ground_truth_image = ground_truth_image.resize(self.resize_target[::-1])
            preprocessed_input_image = self.preprocess(input_image)
            preprocessed_ground_truth_image = self.preprocess(ground_truth_image)
            start_time = time()
            model_output = self.model(preprocessed_input_image)
            inference_time = time() - start_time
            total_inference_time += inference_time
            model_output = model_output.numpy()
            metric_results = []
            for idx, metric in enumerate(self.metrics):
                metric_value = (
                    metric(preprocessed_ground_truth_image, model_output).numpy().item()
                )
                metric_results.append(metric_value)
                total_metric_values[idx] += metric_value
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
        mean_metric_values = [
            value / len(input_image_paths) for value in total_metric_values
        ]
        metric_values = {
            split_name + "/" + type(self.metrics[idx]).__name__: metic_value
            for idx, metic_value in enumerate(mean_metric_values)
        }
        metric_values[split_name + "/Inference-Time"] = total_inference_time / len(
            input_image_paths
        )
        return metric_values

    def evaluate(self):
        log_dict = {}

        for split_name, (
            input_image_paths,
            ground_truth_image_paths,
        ) in self.image_paths.items():
            metric_values = self.evaluate_split(
                input_image_paths, ground_truth_image_paths, split_name
            )
            log_dict = {**log_dict, **metric_values}
        log_dict["Evaluation"] = self.wandb_table

        trainable_parameters = (
            count_params(self.model._collected_trainable_weights)
            if hasattr(self.model, "_collected_trainable_weights")
            else count_params(self.model.trainable_weights)
        )
        non_trainable_parameters = count_params(self.model.non_trainable_weights)

        log_dict["Trainable Parameters"] = trainable_parameters
        log_dict["Non-Trainable Parameters"] = non_trainable_parameters
        log_dict["Total Parameters"] = trainable_parameters + non_trainable_parameters

        if self.input_size is not None:
            try:
                log_dict["GFLOPs"] = calculate_gflops(
                    model=self.model, input_shape=[self.input_size, self.input_size, 3]
                )
            except:
                pass

        if wandb.run is not None:
            wandb.log(log_dict)
