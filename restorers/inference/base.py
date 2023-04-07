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
    """Abstract base class for building Inferers for different tasks or models.

    - The inferer can perform inference on a single image or on a directory of images.
    - The inferred images are saved to the specified output directory.
    - The inferer can also log the inference time and the input and output images to
        Weights & Biases.

    Abstract functions to be overriden are:

    - `preprocess(self, image: Image) -> Union[np.ndarray, tf.Tensor]`
        - Add custom preprocessing logic that would preprocess a `PIL.Image` and add
            a batch dimension. This function should return a `np.ndarray` or a `tf.Tensor`
            that would be consumed by the model.

    - `postprocess(self, model_output: np.ndarray) -> Image`
        - Add postprocessing logic that would convert the output of the model to a
            `PIL.Image`.

    Args:
        model (Optional[tf.keras.Model]): The model that is to be evaluated. Note that passing
            the model during initializing the evaluator is not compulsory. The model can also
            be set using the function `initialize_model_from_wandb_artifact`.
        resize_factor (Optional[int]): The factor by which the input image should be resized
            for inference.
        model_alias (Optional[str]): The alias of the model that is to be logged to
            Weights & Biases. This is useful for qualitative comparison of results of multiple
            models.
    """

    def __init__(
        self,
        model: Optional[tf.keras.Model] = None,
        resize_factor: Optional[int] = 1,
        model_alias: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.resize_factor = resize_factor
        self.model_alias = model_alias
        self.create_wandb_table()

    @abstractmethod
    def preprocess(self, image: Image) -> Union[np.ndarray, tf.Tensor]:
        """This is an abstract method that would hold the custom preprocessing logic that would
        preprocess a `PIL.Image` and add a batch dimension.

        Args:
            image (PIL.Image): A PIL Image.

        Returns:
            (Union[np.ndarray, tf.Tensor]): A numpy or Tensorflow tensor that would be fed to
                the model.
        """
        raise NotImplementedError(f"{self.__class__.__name__ }.preprocess")

    @abstractmethod
    def postprocess(self, model_output: np.ndarray) -> Image:
        """This is an abstract method that would hold the custom postprocessing logic that
        would convert the output of the model to a `PIL.Image`.

        Args:
            model_output (np.ndarray): Output of the model.

        Returns:
            (PIL.Image): The model output postprocessed to a PIL Image.
        """
        raise NotImplementedError(f"{self.__class__.__name__ }.postprocess")

    def initialize_model_from_wandb_artifact(self, artifact_address: str) -> None:
        """Initialize a `tf.keras.Model` that is to be evaluated from a
        [Weights & Biases artifact](https://docs.wandb.ai/guides/artifacts).

        Args:
            artifact_address (str): Address to the Weights & Biases artifact hosting the model to be
                evaluated.
        """
        self.model_path = fetch_wandb_artifact(artifact_address, artifact_type="model")
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

    def create_wandb_table(self):
        columns = ["Input-Image", "Enhanced-Image", "Inference-Time"]
        columns = columns + ["Model-Alias"] if self.model_alias is not None else columns
        self.wandb_table = wandb.Table(columns=columns)

    def _infer_on_single_image(self, input_path: str, output_path: str):
        input_image = Image.open(input_path).convert("RGB")
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
        table_data = [
            wandb.Image(input_image),
            wandb.Image(post_processed_image),
            inference_time,
        ]
        table_data = (
            table_data + [self.model_alias]
            if self.model_alias is not None
            else table_data
        )
        self.wandb_table.add_data(*table_data)

    def infer(self, input_path: str, output_path: Optional[str] = None) -> None:
        """Perform inference on a single image or a directory of images. The images are logged
        to a Weights & Biases table in case it is called in the context of a
        [run](https://docs.wandb.ai/guides/runs).

        Args:
            input_path (str): Path to the input image or directory of images.
            output_path (Optional[str]): Path to the output directory where the enhanced images.
        """
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
