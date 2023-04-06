import tensorflow as tf

from .zero_dce import ZeroDCE
from .dce_layer import FastDeepCurveEstimationLayer


class FastZeroDce(ZeroDCE):
    """A faster version of the Zero-DCE (Zero-DCE++) model implemented as a `tf.keras.Model`.

    Zero-reference deep curve estimation is a method for unsupervised low-light image enhancement
    that utilizes a deep learning model to estimate the enhancement curve for an image without any
    reference to the original, well-lit image. The model is trained on a dataset of low-light and
    normal-light images, and learns to predict the enhancement curve that will best improve the visual
    quality of the low-light image. Once the enhancement curve is estimated, it can be applied to the
    low-light image to enhance its visibility. This approach allows for real-time enhancement of
    low-light images without the need for a reference image, making it useful in situations where a
    reference image is not available or impractical to obtain.

    Reference:

    1. [Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation](https://li-chongyi.github.io/Proj_Zero-DCE++.html)
    2. https://github.com/Li-Chongyi/Zero-DCE_extension

    Args:
        num_intermediate_filters (int): number of filters in the intermediate convolutional layers.
        num_iterations (int): number of iterations of enhancement.
        decoder_channel_factor (int): factor by which number filters in the decoder of deep curve
            estimation layer is multiplied.
    """

    def __init__(
        self,
        num_intermediate_filters: int,
        num_iterations: int,
        decoder_channel_factor: int,
        *args,
        **kwargs
    ):
        super().__init__(
            num_intermediate_filters,
            num_iterations,
            decoder_channel_factor,
            *args,
            **kwargs
        )
        self.deep_curve_estimation = FastDeepCurveEstimationLayer(
            num_intermediate_filters=self.num_intermediate_filters,
            num_iterations=self.num_iterations,
            decoder_channel_factor=self.decoder_channel_factor,
        )

    def get_enhanced_image(self, data, output):
        enhanced_image = data
        for idx in range(self.num_iterations):
            enhanced_image = enhanced_image + output * (
                tf.square(enhanced_image) - enhanced_image
            )
        return enhanced_image
