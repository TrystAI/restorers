from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
import tensorflow as tf

from .base import BaseInferer


class LowLightInferer(BaseInferer):
    """Inferer for low-light enhancement models.

    Usage:

    ```py
    import os
    import wandb
    from restorers.inference import LowLightInferer

    # initialize a wandb run for inference
    wandb.init(project="low-light-enhancement", job_type="inference")

    # initialize the inferer
    inferer = LowLightInferer(
    resize_factor=1, model_alias="Zero-DCE"
    )
    # intialize the model from wandb artifacts
    inferer.initialize_model_from_wandb_artifact(
    # This artifact address corresponds to a Zero-DCE model trained on the LoL dataset
    "ml-colabs/low-light-enhancement/run_oaa25znm_model:v99"
    )
    # infer on a directory of images
    # inferer.infer("./dark_images")
    # or infer on a single image
    inferer.infer(sample_image)
    ```

    ??? example "Examples"
        - [Inferece on your own images for low-light enhancement](../../examples/inference_low_light).

    Args:
        model (Optional[tf.keras.Model]): The model that is to be perform inference. Note that
            passing the model during initializing the evaluator is not compulsory. The model can
            also be set using the function `initialize_model_from_wandb_artifact`.
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
        super().__init__(model, resize_factor, model_alias)

    def preprocess(self, image: Image) -> Union[np.ndarray, tf.Tensor]:
        """Preprocessing logic for preprocessing a `PIL.Image` and adding a batch dimension.

        Args:
            image (PIL.Image): A PIL Image.

        Returns:
            (Union[np.ndarray, tf.Tensor]): A numpy or Tensorflow tensor that would be fed to
                the model.
        """
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)

    def postprocess(self, model_output: np.ndarray) -> Image:
        """Postprocessing logic for converting the output of the model to a `PIL.Image`.

        Args:
            model_output (np.ndarray): Output of the model.

        Returns:
            (PIL.Image): The model output postprocessed to a PIL Image.
        """
        model_output = model_output * 255.0
        model_output = model_output.clip(0, 255)
        image = model_output[0].reshape(
            (np.shape(model_output)[1], np.shape(model_output)[2], 3)
        )
        return Image.fromarray(np.uint8(image))
