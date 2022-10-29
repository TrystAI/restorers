import tensorflow as tf

from .mrb import MultiScaleResidualBlock


class RecursiveResidualGroup(tf.keras.layers.Layer):
    def __init__(
        self,
        channels: int,
        num_mrb_blocks: int,
        channel_factor: float,
        groups: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        layers = [
            MultiScaleResidualBlock(channels, channel_factor, groups)
            for _ in range(num_mrb_blocks)
        ]
        layers.append(
            tf.keras.layers.Conv2D(channels, kernel_size=3, strides=1, padding="same")
        )
        self.layers = tf.keras.Sequential(layers)

    def call(self, inputs, *args, **kwargs):
        residual = self.layers(inputs)
        residual = residual + inputs
        return residual
