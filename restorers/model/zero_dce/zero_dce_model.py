from typing import Dict, Any

import tensorflow as tf

from .dce_layer import DeepCurveEstimationLayer
from ...losses import SpatialConsistencyLoss
from ...losses.zero_reference import (
    color_constancy_loss,
    exposure_loss,
    illumination_smoothness_loss,
)


class ZeroDCE(tf.keras.Model):
    def __init__(self, filters: int, num_iterations: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.num_iterations = num_iterations

        self.dce_layer = DeepCurveEstimationLayer(
            filters=self.filters, num_iterations=self.num_iterations
        )

    def compile(
        self,
        weight_color_constancy_loss: float,
        weight_exposure_loss: float,
        weight_illumination_smoothness_loss: float,
        *args,
        **kwargs
    ):
        super().compile(*args, **kwargs)
        self.weight_color_constancy_loss = weight_color_constancy_loss
        self.weight_exposure_loss = weight_exposure_loss
        self.weight_illumination_smoothness_loss = weight_illumination_smoothness_loss
        self.spatial_constancy_loss = SpatialConsistencyLoss()

    def get_enhanced_image(self, data, curve_parameter_maps):
        enhanced_image = data
        curve_parameter_maps = tf.split(
            curve_parameter_maps, num_or_size_splits=self.num_iterations, axis=-1
        )
        for curve_parameter_map in curve_parameter_maps:
            enhanced_image = enhanced_image + curve_parameter_map * (
                tf.square(enhanced_image) - enhanced_image
            )
        return enhanced_image

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        curve_parameter_maps = self.dce_layer(inputs)
        enhanced_image = self.get_enhanced_image(inputs, curve_parameter_maps)
        return enhanced_image

    def compute_losses(self, data, output):
        enhanced_image = self.get_enhanced_image(data, output)
        loss_illumination = (
            self.weight_illumination_smoothness_loss
            * illumination_smoothness_loss(output)
        )
        loss_spatial_constancy = tf.reduce_mean(
            self.spatial_constancy_loss(enhanced_image, data)
        )
        loss_color_constancy = self.weight_color_constancy_loss * tf.reduce_mean(
            color_constancy_loss(enhanced_image)
        )
        loss_exposure = self.weight_exposure_loss * tf.reduce_mean(
            exposure_loss(enhanced_image)
        )
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
            "color_constancy_loss": loss_color_constancy,
            "exposure_loss": loss_exposure,
        }

    def train_step(self, data):
        with tf.GradientTape() as tape:
            output = self.dce_layer(data)
            losses = self.compute_losses(data, output)
        gradients = tape.gradient(
            losses["total_loss"], self.dce_layer.trainable_weights
        )
        self.optimizer.apply_gradients(zip(gradients, self.dce_layer.trainable_weights))
        return losses

    def test_step(self, data):
        output = self.dce_layer(data)
        return self.compute_losses(data, output)

    def get_config(self) -> Dict[str, Any]:
        return {
            "filters": self.filters,
            "num_iterations": self.num_iterations,
        }
