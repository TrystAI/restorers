import os
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Callable, Dict

import wandb
import numpy as np
from tqdm import tqdm
from PIL import Image

import tensorflow as tf


class BaseInferer(ABC):
    """Inference Class for an MirNet Model.

    Args:
        model (Optional[keras.Model], optional):
            The `tf.keras.Model` to be used for inference. Defaults to None.
            If None, the address of a Weights & Biases model artifact should be passed.
        model_artifact_address (Optional[str], optional):
            The address of the model artifact on Weights & Biases or a URL. Defaults to None.
            If None, a `tf.keras.Model` should be passed to `model`.
        psnr_max_val (float):
            The dynamic range of the pixel values in the images for calculation of
            peak signal-noise ratio. Default is 1.0.
        ssim_max_val (float):
            The dynamic range of the pixel values in the images for calculation of
            structural similarity. Default is 1.0.

    Usage:
        A `tf.keras.Model` can be passed to the `model` parameter.

        ```python
        model = tf.keras.models.load_model("path/to/model")

        inferer = Inferer(model=model)
        ```

        Optinally the address of a Weights & Biases model artifact
        can be passed to the parameter`model_artifact_address`.

        ```python
        inferer = Inferer(
            model_artifact_address="ml-colabs/compressed-mirnet/run_1poazfk5_model:latest"
        )
        ```

        Note:
            A Weights & Biases run is not necessary to be initialized
            in order to use a model artifact. However, if a run is initialized,
            the inference operation will be tracked in the lineage of the artifacts being used.
    """

    def __init__(
        self,
        model: Optional[tf.keras.Model] = None,
        model_artifact_address: Optional[str] = None,
        metrics: Dict[str, Callable] = {},
    ) -> None:
        self.model = model
        self.model_artifact_address = model_artifact_address
        self.metrics = metrics
        if self.model is None:
            self.initialize_model()

    def initialize_model(self) -> None:
        artifact = (
            wandb.Api().artifact(self.model_artifact_address, type="model")
            if wandb.run is None
            else wandb.use_artifact(self.model_artifact_address, type="model")
        )
        model_path = artifact.download()
        producer_run = artifact.logged_by()
        self.model_configs = producer_run.config["model_configs"]
        self.model = tf.keras.models.load_model(model_path, compile=False)

    @abstractmethod
    def preprocess_image(self, image: Image):
        raise NotImplementedError(f"{self.__class__.__name__ }.preprocess_image")

    @abstractmethod
    def postprocess_image(self, model_output: Union[np.array, tf.Tensor]):
        raise NotImplementedError(f"{self.__class__.__name__ }.postprocess_image")

    def single_image_file_inference(self, image_file):
        input_image = Image.open(image_file)
        preprocessed_image = self.preprocess_image(input_image)
        model_output = self.model.predict(preprocessed_image, verbose=0)
        predicted_image = self.postprocess_image(model_output)[0]
        return input_image, model_output, predicted_image

    def single_image_batch_inference(self, image_batch):
        input_images = [Image.fromarray(np.uint8(image)) for image in image_batch]
        preprocessed_low_light_images_batch = input_images.astype("float32") / 255.0
        model_output_batch = self.model.predict(
            preprocessed_low_light_images_batch, verbose=0
        )
        post_processed_images = self.postprocess_image(model_output_batch)
        return input_images, model_output_batch, post_processed_images
