import tensorflow as tf

from .dce_net import DeepCurveEstimationNetwork
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

        self.dce_net = DeepCurveEstimationNetwork(
            num_intermediate_filters=self.num_intermediate_filters,
            num_iterations=self.num_iterations,
        )

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.spatial_constancy_loss = SpatialConsistencyLoss()

    def get_enhanced_image(self, data, output):
        r1 = output[:, :, :, :3]
        r2 = output[:, :, :, 3:6]
        r3 = output[:, :, :, 6:9]
        r4 = output[:, :, :, 9:12]
        r5 = output[:, :, :, 12:15]
        r6 = output[:, :, :, 15:18]
        r7 = output[:, :, :, 18:21]
        r8 = output[:, :, :, 21:24]
        x = data + r1 * (tf.square(data) - data)
        x = x + r2 * (tf.square(x) - x)
        x = x + r3 * (tf.square(x) - x)
        enhanced_image = x + r4 * (tf.square(x) - x)
        x = enhanced_image + r5 * (tf.square(enhanced_image) - enhanced_image)
        x = x + r6 * (tf.square(x) - x)
        x = x + r7 * (tf.square(x) - x)
        enhanced_image = x + r8 * (tf.square(x) - x)
        return enhanced_image

    def call(self, data):
        dce_net_output = self.dce_net(data)
        return self.get_enhanced_image(data, dce_net_output)

    def compute_losses(self, data, output):
        enhanced_image = self.get_enhanced_image(data, output)
        loss_illumination = 200 * illumination_smoothness_loss(output)
        loss_spatial_constancy = tf.reduce_mean(
            self.spatial_constancy_loss(enhanced_image, data)
        )
        loss_color_constancy = 5 * tf.reduce_mean(color_constancy_loss(enhanced_image))
        loss_exposure = 10 * tf.reduce_mean(exposure_loss(enhanced_image))
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
            output = self.dce_net(data)
            losses = self.compute_losses(data, output)
        gradients = tape.gradient(losses["total_loss"], self.dce_net.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.dce_net.trainable_weights))
        return losses

    def test_step(self, data):
        output = self.dce_net(data)
        return self.compute_losses(data, output)
