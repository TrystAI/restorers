import os
import shutil
import tempfile
from typing import Optional, Union, List, Dict, Callable

import wandb
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm.autonotebook import tqdm

from .base import BaseInferer
from ..model import MirNetv2


class LowLightInferer(BaseInferer):
    def __init__(
        self,
        model: Optional[tf.keras.Model] = None,
        model_artifact_address: Optional[str] = None,
        metrics: Dict[str, Callable] = {},
    ) -> None:
        super().__init__(model, model_artifact_address, metrics)

    def initialize_model(self):
        super().initialize_model()
        temp_dir = tempfile.mkdtemp()
        weights_dir = os.path.join(temp_dir, "model_weights")
        self.model.save_weights(weights_dir)
        self.model = model = MirNetv2(
            channels=self.model_configs["channels"],
            channel_factor=self.model_configs["channel_factor"],
            num_mrb_blocks=self.model_configs["num_mrb_blocks"],
            add_residual_connection=self.model_configs["add_residual_connection"],
        )
        self.model.load_weights(weights_dir)
        shutil.rmtree(temp_dir)

    def preprocess_image(self, image: Image):
        """Preprocesses the image for inference.
        Returns:
            A numpy array of shape (1, height, width, 3) preprocessed for inference.
        """
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)

    def postprocess_image(self, model_output: Union[np.array, tf.Tensor]):
        """Postprocesses the model output for inference.
        Returns:
            A list of PIL.Image.Image objects postprocessed for visualization.
        """
        model_output = model_output * 255.0
        model_output = model_output.clip(0, 255)
        images = []
        for idx in range(model_output.shape[0]):
            image = model_output[idx].reshape(
                (np.shape(model_output)[1], np.shape(model_output)[2], 3)
            )
            images.append(Image.fromarray(np.uint8(image)))
        return images
