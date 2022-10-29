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


class MirNetv2(tf.keras.Model):
    def __init__(
        self,
        channels: int,
        channel_factor: float,
        num_mrb_blocks: int,
        add_residual_connection: bool,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.add_residual_connection = add_residual_connection

        self.conv_in = tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same")

        self.rrg_blocks = tf.keras.Sequential(
            [
                RecursiveResidualGroup(
                    channels, num_mrb_blocks, channel_factor, groups=1
                ),
                RecursiveResidualGroup(
                    channels, num_mrb_blocks, channel_factor, groups=2
                ),
                RecursiveResidualGroup(
                    channels, num_mrb_blocks, channel_factor, groups=4
                ),
                RecursiveResidualGroup(
                    channels, num_mrb_blocks, channel_factor, groups=4
                ),
            ]
        )

        self.conv_out = tf.keras.layers.Conv2D(3, kernel_size=3, padding="same")

    def call(self, inputs, training=None, mask=None):
        shallow_features = self.conv_in(inputs)
        deep_features = self.rrg_blocks(shallow_features)
        output = self.conv_out(deep_features)
        output = output + inputs if self.add_residual_connection else output
        return output
