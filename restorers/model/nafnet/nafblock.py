from typing import Optional

import tensorflow as tf
from tensorflow import keras

from .baseline import BaselineBlock


class SimpleGate(keras.layers.Layer):
    """
    Simple Gate
    It splits the input of size (b,h,w,c) into tensors of size (b,h,w,c//factor) and returns their Hadamard product
    Parameters:
        factor: the amount by which the channels are scaled down
    """

    def __init__(self, factor: Optional[int] = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        x = tf.expand_dims(x, axis=-1)
        return tf.reduce_prod(
            tf.concat(tf.split(x, num_or_size_splits=self.factor, axis=-2), axis=-1),
            axis=-1,
        )

    def get_config(self) -> dict:
        """Add factor to the config"""
        config = super().get_config()
        config.update({"factor": self.factor})
        return config


class SimplifiedChannelAttention(keras.layers.Layer):
    """
    Simplified Channel Attention layer
    It is a modification of channel attention without any non-linear activations.
    Parameters:
        channels: number of channels in input
    """

    def __init__(self, channels: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.conv = keras.layers.Conv2D(filters=channels, kernel_size=1)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        average_pooling = self.avg_pool(inputs)
        feature_descriptor = tf.reshape(
            average_pooling, shape=(-1, 1, 1, self.channels)
        )
        features = self.conv(feature_descriptor)
        return inputs * features

    def get_config(self) -> dict:
        """Add channels to the config"""
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


class NAFBlock(BaselineBlock):
    """
    NAFBlock (Nonlinear Activation Free Block)

    It is derived from the Baseline Block by removing all the non-linear activations

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
        self.activation = SimpleGate(factor)

    def get_dw_channel(self, input_channels: int) -> int:
        return input_channels * self.factor

    def get_attention_layer(self, input_shape: tf.TensorShape) -> None:
        input_channels = input_shape[-1]
        return SimplifiedChannelAttention(input_channels)
