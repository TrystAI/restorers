from time import time
from PIL import Image
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import wandb
import numpy as np
import tensorflow as tf

from ..utils import fetch_wandb_artifact


class BaseInferer(ABC):
    def __init__(
        self,
        model: Optional[tf.keras.Model] = None,
        resize_target: Optional[Tuple[int, int]] = None,
    ):
        self.model = model
        self.resize_target = resize_target
        self.wandb_table = (
            wandb.Table(columns=["Input-Image", "Enhaned-Image", "Inference-Time"])
            if wandb.run is not None
            else None
        )

    @abstractmethod
    def preprocess(self, image_path: Image) -> Union[np.ndarray, tf.Tensor]:
        raise NotImplementedError(f"{self.__class__.__name__ }.preprocess")

    @abstractmethod
    def postprocess(self, model_output: np.ndarray) -> Image:
        raise NotImplementedError(f"{self.__class__.__name__ }.postprocess")

    def initialize_model_from_wandb_artifact(self, artifact_address: str) -> None:
        model_path = fetch_wandb_artifact(artifact_address, artifact_type="model")
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def infer(self, input_image_path: str, output_path: Optional[str] = None):
        input_image = Image.open(input_image_path)
        input_image = (
            input_image.resize(self.resize_target[::-1])
            if self.resize_target
            else input_image
        )
        preprocessed_input_image = self.preprocess(input_image)
        start_time = time()
        model_output = self.model.predict(preprocessed_input_image, verbose=0)
        inference_time = time() - start_time
        post_processed_image = self.postprocess(model_output)
        if output_path is not None:
            post_processed_image.save(output_path)
        self.wandb_table.add_data(input_image, post_processed_image, inference_time)
        wandb.log({"Inference": self.wandb_table})
