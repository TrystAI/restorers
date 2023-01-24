from typing import Dict

import tensorflow as tf

from .dce_layer import DeepCurveEstimationLayer
from restorers.losses import SpatialConsistencyLoss
from restorers.losses.zero_reference import (
    color_constancy,
    exposure_control_loss,
    illumination_smoothness_loss,
)


class ZeroDCE(tf.keras.Model):
    def __init__(
        self, num_intermediate_filters: int, num_iterations: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.num_intermediate_filters = num_intermediate_filters
        self.num_iterations = num_iterations

        self.deep_curve_estimation = DeepCurveEstimationLayer(
            num_intermediate_filters=self.num_intermediate_filters,
            num_iterations=self.num_iterations,
        )

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.spatial_constancy_loss = SpatialConsistencyLoss()

    def get_enhanced_image(self, data, output):
        curves = tf.split(output, self.num_iterations, axis=-1)
        enhanced_image = data
        for idx in range(self.num_iterations):
            enhanced_image = enhanced_image + curves[idx] * (
                tf.square(enhanced_image) - enhanced_image
            )
        return enhanced_image

    def call(self, data):
        dce_net_output = self.deep_curve_estimation(data)
        return self.get_enhanced_image(data, dce_net_output)

    def compute_losses(self, data, output):
        enhanced_image = self.get_enhanced_image(data, output)
        loss_illumination = 200 * illumination_smoothness_loss(output)
        loss_spatial_constancy = tf.reduce_mean(
            self.spatial_constancy_loss(enhanced_image, data)
        )
        loss_color_constancy = 5 * tf.reduce_mean(color_constancy(enhanced_image))
        loss_exposure = 10 * tf.reduce_mean(exposure_control_loss(enhanced_image))
        total_loss = (
            loss_illumination
            + loss_spatial_constancy
            + loss_color_constancy
            + loss_exposure
        )
        return {
            "total_loss": total_loss,
            "illumination_smoothness_loss": loss_illumination,
            "spatial_constancy_loss": loss_spatial_constancy,
            "color_constancy": loss_color_constancy,
            "exposure_control_loss": loss_exposure,
        }

    def train_step(self, data):
        with tf.GradientTape() as tape:
            output = self.deep_curve_estimation(data)
            losses = self.compute_losses(data, output)
        gradients = tape.gradient(losses["total_loss"], self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return losses

    def test_step(self, data):
        output = self.deep_curve_estimation(data)
        return self.compute_losses(data, output)

    def get_config(self) -> Dict:
        return {
            "num_intermediate_filters": self.num_intermediate_filters,
            "num_iterations": self.num_iterations,
        }

    def save(self, filepath, *args, **kwargs):
        input_tensor = tf.keras.Input(shape=[None, None, 3])
        saved_model = tf.keras.Model(
            inputs=input_tensor, outputs=self.call(input_tensor)
        )
        saved_model.save(filepath, *args, **kwargs)
