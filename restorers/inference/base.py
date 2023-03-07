import os
from glob import glob
from time import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import wandb
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm.auto import tqdm

from ..utils import fetch_wandb_artifact


class BaseInferer(ABC):
    def __init__(
        self,
        model: Optional[tf.keras.Model] = None,
        resize_factor: Optional[int] = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.resize_factor = resize_factor
        self.wandb_table = wandb.Table(
            columns=["Input-Image", "Enhanced-Image", "Inference-Time"]
        )

    @abstractmethod
    def preprocess(self, image_path: Image) -> Union[np.ndarray, tf.Tensor]:
        raise NotImplementedError(f"{self.__class__.__name__ }.preprocess")

    @abstractmethod
    def postprocess(self, model_output: np.ndarray) -> Image:
        raise NotImplementedError(f"{self.__class__.__name__ }.postprocess")
    
    def initialize_model_from_wandb_artifact(self, artifact_address: str) -> None:
        self.model_path = fetch_wandb_artifact(artifact_address, artifact_type="model")
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

    def _infer_on_single_image(self, input_path: str, output_path: str):
        input_image = Image.open(input_path).convert('RGB')
        if self.resize_factor > 1:
            width, height = input_image.size
            width = (width // self.resize_factor) * self.resize_factor
            height = (height // self.resize_factor) * self.resize_factor
            input_image = input_image.resize((width, height))
        preprocessed_input_image = self.preprocess(input_image)
        start_time = time()
        model_output = self.model(preprocessed_input_image)
        inference_time = time() - start_time
        post_processed_image = self.postprocess(model_output.numpy())
        if output_path is not None:
            post_processed_image.save(post_processed_image)
        self.wandb_table.add_data(
            wandb.Image(input_image), wandb.Image(post_processed_image), inference_time
        )

    def infer(self, input_path: str, output_path: Optional[str] = None):
        if os.path.isdir(input_path):
            input_images = glob(os.path.join(input_path, "*"))
            for input_image_path in tqdm(input_images):
                output_path = (
                    os.path.join(output_path, os.path.basename(input_image_path))
                    if output_path is not None
                    else None
                )
                self._infer_on_single_image(
                    input_path=input_image_path, output_path=output_path
                )
        else:
            self._infer_on_single_image(input_path=input_path, output_path=output_path)
        if wandb.run is not None:
            wandb.log({"Inference": self.wandb_table})
