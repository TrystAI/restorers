import os
from glob import glob
from typing import Dict, Callable, Optional

import wandb
import numpy as np
import tensorflow as tf

from .base import BaseEvaluator
from ..dataloader.base.commons import read_image
from ..utils import scale_tensor, fetch_wandb_artifact


class LoLEvaluator(BaseEvaluator):
    def __init__(
        self,
        metrics: Dict[str, Callable],
        model: Optional[tf.keras.Model] = None,
        bit_depth: float = 8,
        benchmark_against_input: bool = False,
    ):
        self.normalization_factor = (2**bit_depth) - 1
        self.dataset_artifact_address = "ml-colabs/dataset/LoL:v0"
        self.benchmark_against_input = benchmark_against_input
        super().__init__(metrics, model)

    def preprocess(self, image_path):
        return tf.expand_dims(read_image(image_path, self.normalization_factor), axis=0)

    def postprocess(self, input_tensor):
        return np.squeeze(scale_tensor(input_tensor))

    def populate_image_paths(self):
        dataset_path = fetch_wandb_artifact(
            self.dataset_artifact_address, artifact_type="dataset"
        )
        train_low_light_images = sorted(
            glob(os.path.join(dataset_path, "our485", "low", "*"))
        )
        train_enhanced_images = sorted(
            glob(os.path.join(dataset_path, "our485", "high", "*"))
        )
        test_low_light_images = sorted(
            glob(os.path.join(dataset_path, "eval15", "low", "*"))
        )
        test_enhanced_images = sorted(
            glob(os.path.join(dataset_path, "eval15", "high", "*"))
        )
        return (
            {
                "train": (train_low_light_images, train_enhanced_images),
                "eval15": (test_low_light_images, test_enhanced_images),
            }
            if not self.benchmark_against_input
            else {
                "train": (train_low_light_images, train_low_light_images),
                "eval15": (test_low_light_images, test_low_light_images),
            }
        )
