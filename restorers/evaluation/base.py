from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Dict, Callable, Optional

import wandb
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from ..utils import fetch_wandb_artifact


class BaseEvaluator(ABC):
    def __init__(
        self, metrics: Dict[str, Callable], model: Optional[tf.keras.Model] = None
    ):
        """Base Class for Evaluating an Image Restoration Model"""
        self.model = model
        self.metrics = OrderedDict(metrics)
        self.evaluation_report = {}
        self.image_paths = self.populate_image_paths()
        self.wandb_table = self.create_wandb_table() if wandb.run is not None else None

    @abstractmethod
    def preprocess(self, image_path: str):
        raise NotImplementedError(f"{self.__class__.__name__ }.preprocess")

    @abstractmethod
    def postprocess(self, input_tensor: np.array):
        raise NotImplementedError(f"{self.__class__.__name__ }.postprocess")

    @abstractmethod
    def populate_image_paths(self) -> Dict[str, Dict[str, List[str]]]:
        raise NotImplementedError(f"{self.__class__.__name__ }.populate_image_paths")

    def create_wandb_table(self) -> wandb.Table:
        columns = ["Split", "Input-Image", "Ground-Truth-Image", "Enhanced-Image"]
        columns = columns + [metric_alias for metric_alias, _ in self.metrics.items()]
        return wandb.Table(columns=columns)

    def initialize_model_from_wandb_artifact(self, artifact_address: str) -> None:
        model_path = fetch_wandb_artifact(artifact_address, artifact_type="model")
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def count_params(self, weights) -> int:
        """Count the total number of scalars composing the weights.

        Reference:
            https://github.com/keras-team/keras/blob/e6784e4302c7b8cd116b74a784f4b78d60e83c26/keras/utils/layer_utils.py#L107

        Args:
            weights: An iterable containing the weights on which to compute params

        Returns:
            (int): The total number of scalars composing the weights
        """
        unique_weights = {id(w): w for w in weights}.values()
        # Ignore TrackableWeightHandlers, which will not have a shape defined.
        unique_weights = [w for w in unique_weights if hasattr(w, "shape")]
        weight_shapes = [w.shape.as_list() for w in unique_weights]
        standardized_weight_shapes = [
            [0 if w_i is None else w_i for w_i in w] for w in weight_shapes
        ]
        return int(sum(np.prod(p) for p in standardized_weight_shapes))

    def get_gflops(self) -> float:
        """
        Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
        in inference mode. It uses tf.compat.v1.profiler under the hood.

        Reference:
            [GFLOPs Computation function by Weights & Biases](https://github.com/wandb/wandb/blob/main/wandb/integration/keras/keras.py#L1043-L1089)

        Returns:
            (float): number of GFLOPs in the model
        """
        if not hasattr(self, "model"):
            raise wandb.Error("self.model must be set before using this method.")

        if not isinstance(
            self.model, (tf.keras.models.Sequential, tf.keras.models.Model)
        ):
            raise ValueError(
                "Calculating FLOPS is only supported for "
                "`tf.keras.Model` and `tf.keras.Sequential` instances."
            )

        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
        )

        # Compute FLOPs for one sample
        batch_size = 1
        inputs = [
            tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
            for inp in self.model.inputs
        ]

        # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
        real_model = tf.function(self.model).get_concrete_function(inputs)
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

        # Calculate FLOPs with tf.profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = (
            tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
            )
            .with_empty_output()
            .build()
        )

        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )

        # convert to GFLOPs
        return (flops.total_float_ops / 1e9) / 2

    def evaluate_splits(self):
        for split, (
            input_image_files,
            ground_truth_image_files,
        ) in self.image_paths.items():
            pbar = tqdm(
                zip(input_image_files, ground_truth_image_files),
                total=len(input_image_files),
                desc=f"Evaluating {split} split",
            )
            total_metric_results = [0.0] * len(self.metrics.keys())
            for input_image_file, ground_truth_image_file in pbar:
                preprocessed_input_image = self.preprocess(input_image_file)
                preprocessed_ground_truth_image = self.preprocess(
                    ground_truth_image_file
                )
                model_output = self.model.predict(preprocessed_input_image, verbose=0)
                metric_results = [
                    metric_fn(preprocessed_ground_truth_image, model_output)
                    .numpy()
                    .item()
                    for _, metric_fn in self.metrics.items()
                ]
                total_metric_results = [
                    total_metric_results[idx] + metric_results[idx]
                    for idx in range(len(self.metrics.keys()))
                ]
                postprocessed_input_image = self.postprocess(preprocessed_input_image)
                postprocessed_ground_truth_image = self.postprocess(
                    preprocessed_ground_truth_image
                )
                postprocessed_enhanced_image = self.postprocess(model_output)
                if self.wandb_table is not None:
                    table_data = [
                        split,
                        wandb.Image(postprocessed_input_image),
                        wandb.Image(postprocessed_ground_truth_image),
                        wandb.Image(postprocessed_enhanced_image),
                    ] + metric_results
                    self.wandb_table.add_data(*table_data)
            mean_metric_results = [
                metric_result / len(input_image_files)
                for metric_result in total_metric_results
            ]
            metric_results = {}
            for idx, (key, _) in enumerate(self.metrics.items()):
                self.evaluation_report[f"{split}/{key}"] = mean_metric_results[idx]
            if self.wandb_table is not None:
                self.evaluation_report["Evaluation-Table"] = self.wandb_table

    def evaluate(self):
        trainable_parameters = (
            self.count_params(self.model._collected_trainable_weights)
            if hasattr(self.model, "_collected_trainable_weights")
            else self.count_params(self.model.trainable_weights)
        )
        non_trainable_parameters = self.count_params(self.model.non_trainable_weights)

        self.evaluation_report["GFLOPs"] = self.get_gflops()
        self.evaluation_report["Trainable Parameters"] = trainable_parameters
        self.evaluation_report["Non-Trainable Parameters"] = non_trainable_parameters
        self.evaluation_report["Total Parameters"] = (
            trainable_parameters + non_trainable_parameters
        )

        self.evaluate_splits()

        if wandb.run is not None:
            wandb.log(self.evaluation_report)

        return self.evaluation_report
