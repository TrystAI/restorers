import numpy as np
from PIL import Image
import tensorflow as tf

from .base import BaseInferer


class LowLightInferer(BaseInferer):
    def __init__(
        self,
        model: Optional[tf.keras.Model] = None,
        resize_target: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(model=model, resize_target=resize_target)

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
