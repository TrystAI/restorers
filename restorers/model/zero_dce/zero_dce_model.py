from typing import Dict, Tuple

import tensorflow as tf

from restorers.losses import SpatialConsistencyLoss
from restorers.losses.zero_reference import (
    color_constancy,
    exposure_control_loss,
    illumination_smoothness_loss,
)

from .dce_layer import DeepCurveEstimationLayer, FastDeepCurveEstimationLayer


class ZeroDCE(tf.keras.Model):
    """The Zero-reference Deep Curve Estimation (Zero-DCE) model implemented as a
    `tf.keras.Model`.

    Zero-reference deep curve estimation is a method for unsupervised low-light image enhancement
    that utilizes a deep learning model to estimate the enhancement curve for an image without any
    reference to the original, well-lit image. The model is trained on a dataset of low-light and
    normal-light images, and learns to predict the enhancement curve that will best improve the visual
    quality of the low-light image. Once the enhancement curve is estimated, it can be applied to the
    low-light image to enhance its visibility. This approach allows for real-time enhancement of
    low-light images without the need for a reference image, making it useful in situations where a
    reference image is not available or impractical to obtain.

    Reference:

    1. [Zero-DCE: Zero-reference Deep Curve Estimation for Low-light Image Enhancement](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)
    2. [Zero-Reference Learning for Low-Light Image Enhancement (Supplementary Material)](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Guo_Zero-Reference_Deep_Curve_CVPR_2020_supplemental.pdf)
    3. [Official PyTorch implementation of Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE)
    4. [Unofficial PyTorch implementation of Zero-DCE](https://github.com/bsun0802/Zero-DCE)
    5. [Tensorflow implementation of Zero-DCE](https://github.com/tuvovan/Zero_DCE_TF)
    6. [Keras tutorial for implementing Zero-DCE](https://keras.io/examples/vision/zero_dce/#deep-curve-estimation-model)

    Args:
        num_intermediate_filters (int): number of filters in the intermediate convolutional layers.
        num_iterations (int): number of iterations of enhancement.
    """

    def __init__(
        self, num_intermediate_filters: int, num_iterations: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_intermediate_filters = num_intermediate_filters
        self.num_iterations = num_iterations

        self.deep_curve_estimation = DeepCurveEstimationLayer(
            num_intermediate_filters=self.num_intermediate_filters,
            num_iterations=self.num_iterations,
        )

    def compile(
        self,
        weight_exposure_loss: float,
        weight_color_constancy_loss: float,
        weight_illumination_smoothness_loss: float,
        *args,
        **kwargs
    ) -> None:
        """Configures the model for training.

        Example:

        ```python
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            weight_exposure_loss=1.0,
            weight_color_constancy_loss=0.5,
            weight_illumination_smoothness_loss=20.0,
        )
        ```

        Args:
            weight_exposure_loss (float): weight of the exposure control loss.
            weight_color_constancy_loss (float): weight of the color constancy loss.
            weight_illumination_smoothness_loss (float): weight of the illumination smoothness loss.
        """
        super().compile(*args, **kwargs)
        self.weight_exposure_loss = weight_exposure_loss
        self.weight_color_constancy_loss = weight_color_constancy_loss
        self.weight_illumination_smoothness_loss = weight_illumination_smoothness_loss
        self.spatial_constancy_loss = SpatialConsistencyLoss()

    def get_enhanced_image(
        self, data: tf.Tensor, output: tf.Tensor
    ) -> Tuple[tf.Tensor]:
        curves = tf.split(output, self.num_iterations, axis=-1)
        enhanced_image = data
        for idx in range(self.num_iterations):
            enhanced_image = enhanced_image + curves[idx] * (
                tf.square(enhanced_image) - enhanced_image
            )
        return enhanced_image

    def call(self, data: tf.Tensor, training=None, mask=None) -> Tuple[tf.Tensor]:
        dce_net_output = self.deep_curve_estimation(data)
        return self.get_enhanced_image(data, dce_net_output)

    def compute_losses(
        self, data: tf.Tensor, output: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        enhanced_image = self.get_enhanced_image(data, output)
        loss_illumination = illumination_smoothness_loss(output)
        loss_spatial_constancy = tf.reduce_mean(
            self.spatial_constancy_loss(enhanced_image, data)
        )
        loss_color_constancy = tf.reduce_mean(color_constancy(enhanced_image))
        loss_exposure = tf.reduce_mean(exposure_control_loss(enhanced_image))
        total_loss = (
            loss_spatial_constancy
            + self.weight_illumination_smoothness_loss * loss_illumination
            + self.weight_color_constancy_loss * loss_color_constancy
            + self.weight_exposure_loss * loss_exposure
        )
        return {
            "total_loss": total_loss,
            "illumination_smoothness_loss": loss_illumination,
            "spatial_constancy_loss": loss_spatial_constancy,
            "color_constancy": loss_color_constancy,
            "exposure_control_loss": loss_exposure,
        }

    def train_step(self, data: tf.Tensor) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            output = self.deep_curve_estimation(data)
            losses = self.compute_losses(data, output)
        gradients = tape.gradient(losses["total_loss"], self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return losses

    def test_step(self, data: tf.Tensor) -> Dict[str, tf.Tensor]:
        output = self.deep_curve_estimation(data)
        return self.compute_losses(data, output)

    def get_config(self) -> Dict:
        return {
            "num_intermediate_filters": self.num_intermediate_filters,
            "num_iterations": self.num_iterations,
        }

    def save(self, filepath: str, *args, **kwargs) -> None:
        input_tensor = tf.keras.Input(shape=[None, None, 3])
        saved_model = tf.keras.Model(
            inputs=input_tensor, outputs=self.call(input_tensor)
        )
        saved_model.save(filepath, *args, **kwargs)


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
    """

    def __init__(
        self, num_intermediate_filters: int, num_iterations: int, *args, **kwargs
    ):
        super().__init__(num_intermediate_filters, num_iterations, *args, **kwargs)
        self.deep_curve_estimation = FastDeepCurveEstimationLayer(
            num_intermediate_filters=self.num_intermediate_filters,
            num_iterations=self.num_iterations,
        )

    def get_enhanced_image(self, data, output):
        enhanced_image = data
        for idx in range(self.num_iterations):
            enhanced_image = enhanced_image + output * (
                tf.square(enhanced_image) - enhanced_image
            )
        return enhanced_image
