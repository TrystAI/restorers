import os
from glob import glob
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

from .base import BaseEvaluator
from ..utils import fetch_wandb_artifact


class LoLEvaluator(BaseEvaluator):
    def __init__(
        self,
        metrics: List[tf.keras.metrics.Metric],
        model: Optional[tf.keras.Model] = None,
        input_size: Optional[List[int]] = None,
        resize_target: Optional[Tuple[int, int]] = None,
        dataset_artifact_address: str = None,
    ) -> None:
        """Evaluator for LoL Dataset.

        Args:
            metrics (List[tf.keras.metrics.Metric]): list of keras metrics.
            model (Optional[tf.keras.Model]): model to be evaluated.
            input_size (Optional[List[int]]): input size used for calculating GFLOPs.
            resize_target: (Optional[Tuple[int, int]]): resize to this size for inference.
            dataset_artifact_address (str): address of WandB artifact hosting LoL dataset.
        """
        self.dataset_artifact_address = dataset_artifact_address
        super().__init__(metrics, model, input_size, resize_target)

    def preprocess(self, image: Image) -> Union[np.ndarray, tf.Tensor]:
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)

    def postprocess(self, model_output: np.ndarray) -> Image:
        model_output = model_output * 255.0
        model_output = model_output.clip(0, 255)
        image = model_output[0].reshape(
            (np.shape(model_output)[1], np.shape(model_output)[2], 3)
        )
        return Image.fromarray(np.uint8(image))

    def populate_image_paths(self) -> Dict[str, Tuple[List[str], List[str]]]:
        dataset_path = fetch_wandb_artifact(
            self.dataset_artifact_address, artifact_type="dataset"
        )
        train_low_light_images = sorted(
            glob(os.path.join(dataset_path, "our485", "low", "*"))
        )
        train_ground_truth_images = sorted(
            glob(os.path.join(dataset_path, "our485", "high", "*"))
        )
        test_low_light_images = sorted(
            glob(os.path.join(dataset_path, "eval15", "low", "*"))
        )
        test_ground_truth_images = sorted(
            glob(os.path.join(dataset_path, "eval15", "high", "*"))
        )
        return {
            "Train-Val": (train_low_light_images, train_ground_truth_images),
            "Eval15": (test_low_light_images, test_ground_truth_images),
        }
