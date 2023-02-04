from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Union, List, Dict, Callable, Optional

import PIL
import wandb
import numpy as np
import tensorflow as tf


class BaseEvaluator(ABC):
    def __init__(
        self, metrics: Dict[str, Callable], model: Optional[tf.keras.Model] = None
    ):
        self.model = model
        self.metrics = OrderedDict(metrics)
        self.evaluation_report = {}
        self.image_paths = self.populate_image_paths()
        self.wandb_table = self.create_wandb_table() if wandb.run is not None else None

    @abstractmethod
    def preprocess(self, image_path: str):
        raise NotImplementedError(f"{self.__class__.__name__ }.preprocess")

    @abstractmethod
    def postprocess(self, image_path: str):
        raise NotImplementedError(f"{self.__class__.__name__ }.postprocess")

    @abstractmethod
    def populate_image_paths(self) -> Dict[str, Dict[str, List[str]]]:
        raise NotImplementedError(f"{self.__class__.__name__ }.postprocess")

    def create_wandb_table(self) -> wandb.Table:
        columns = ["Split", "Input-Image", "Ground-Truth-Image", "Enhanced-Image"]
        for key, _ in self.metrics.items():
            columns.append(key)
        return wandb.Table(columns=columns)

    def fetch_model_from_wandb_artifact(self, artifact_address: str) -> None:
        model_path = (
            wandb.Api()
            .artifact(self.dataset_artifact_address, type="dataset")
            .download()
            if wandb.run is None
            else wandb.use_artifact(
                self.dataset_artifact_address, type="model"
            ).download()
        )
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

    def report_results(self):
        trainable_parameters = (
            count_params(self.model._collected_trainable_weights)
            if hasattr(model, "_collected_trainable_weights")
            else count_params(self.model.trainable_weights)
        )
        non_trainable_parameters = count_params(self.model.non_trainable_weights)

        self.evaluation_report["GFLOPs"] = self.get_gflops()
        self.evaluation_report["Trainable Parameters"] = trainable_parameters
        self.evaluation_report["Non-Trainable Parameters"] = non_trainable_parameters
        self.evaluation_report["Total Parameters"] = (
            trainable_parameters + non_trainable_parameters
        )

        if wandb.run is not None:
            wandb.log(self.evaluation_report)

        return self.evaluation_report
