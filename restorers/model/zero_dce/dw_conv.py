from typing import Dict

import tensorflow as tf


class DepthwiseSeparableConvolution(tf.keras.layers.Layer):
    """Depthwise-separable convolution implemented as a `tf.keras.layers.Layer`.

    Reference:

    1. [Official PyTorch implementation of Zero-DCE++](https://github.com/Li-Chongyi/Zero-DCE_extension/blob/main/Zero-DCE%2B%2B/model.py#L8)

    Args:
        input_channels (int): number of input channels.
        output_channels (int): number of output channels.
    """

    def __init__(
        self, intermediate_channels: int, output_channels: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.intermediate_channels = intermediate_channels
        self.output_channels = output_channels

        self.depthwise_convolution = tf.keras.layers.Conv2D(
            filters=intermediate_channels,
            kernel_size=(3, 3),
            padding="same",
            groups=intermediate_channels,
        )
        self.pointwise_convolution = tf.keras.layers.Conv2D(
            filters=output_channels, kernel_size=(1, 1), padding="valid"
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.pointwise_convolution(self.depthwise_convolution(inputs))

    def get_config(self) -> Dict:
        return {
            "intermediate_channels": self.intermediate_channels,
            "output_channels": self.output_channels,
        }
