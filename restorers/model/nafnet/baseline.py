from typing import Optional

import tensorflow as tf
from tensorflow import keras

from .plainnet import PlainBlock


class ChannelAttention(keras.layers.Layer):
    """
    Channel Attention layer

    Parameters:
        channels: number of channels in input
    """

    def __init__(self, channels: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.conv1 = keras.layers.Conv2D(
            filters=channels // 2, kernel_size=1, activation=keras.activations.relu
        )
        self.conv2 = keras.layers.Conv2D(
            filters=channels, kernel_size=1, activation=keras.activations.sigmoid
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        average_pooling = self.avg_pool(inputs)
        feature_descriptor = tf.reshape(
            average_pooling, shape=(-1, 1, 1, self.channels)
        )
        x = self.conv1(feature_descriptor)
        return inputs * self.conv2(x)

    def get_config(self) -> dict:
        """Add channels to the config"""
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


class BaselineBlock(PlainBlock):
    """
    BaselineBlock Layer

    This is the baseline block proposed in NAFNet Paper.
    The baseline block is derived from plainblock by adding layer norm, channel attention,
     and replacing relu with gelu.

    Parameters:
        input_channels: number of channels in the input (as NAFBlock retains the input size in the output)
        factor: factor by which the channels must be increased before being reduced by simple gate.
            (Higher factor denotes higher order polynomial in multiplication. Default factor is 2)
        drop_out_rate: dropout rate
        balanced_skip_connection: adds additional trainable parameters to the skip connections.
            The parameter denotes how much importance should be given to the sub block in the skip connection.
    """

    def __init__(
        self,
        factor: Optional[int] = 2,
        drop_out_rate: Optional[float] = 0.0,
        balanced_skip_connection: Optional[bool] = False,
        **kwargs
    ) -> None:
        super().__init__(factor, drop_out_rate, balanced_skip_connection, **kwargs)

        self.layer_norm1 = keras.layers.LayerNormalization()
        self.layer_norm2 = keras.layers.LayerNormalization()

        self.activation = keras.layers.Activation("gelu")

    def get_attention_layer(self, input_shape: tf.TensorShape) -> None:
        input_channels = input_shape[-1]
        return ChannelAttention(input_channels)

    def call_block1(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.layer_norm1(inputs)
        return super().call_block1(x)

    def call_block2(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.layer_norm2(inputs)
        return super().call_block2(x)
