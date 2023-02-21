from typing import Optional

import tensorflow as tf
from tensorflow import keras


class PlainBlock(keras.layers.Layer):
    """
    PlainBlock Layer

    This is the plain block proposed in NAFNet Paper.
    This is inspired from the restormer's block, keeping the most common components.
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
        super().__init__(**kwargs)
        self.factor = factor
        self.drop_out_rate = drop_out_rate
        self.balanced_skip_connection = balanced_skip_connection

        self.dropout1 = keras.layers.Dropout(drop_out_rate)
        self.dropout2 = keras.layers.Dropout(drop_out_rate)

    def build(self, input_shape: tf.TensorShape) -> None:
        input_channels = input_shape[-1]

        self.conv1 = keras.layers.Conv2D(
            filters=input_channels, kernel_size=1, strides=1
        )
        self.dconv2 = keras.layers.Conv2D(
            filters=input_channels,
            kernel_size=1,
            padding="same",
            strides=1,
            groups=input_channels,
            activation="relu",
        )

        self.conv3 = keras.layers.Conv2D(
            filters=input_channels, kernel_size=1, strides=1
        )

        self.channel_attention = ChannelAttention(input_channels)

        ffn_channel = input_channels * self.factor

        self.conv4 = keras.layers.Conv2D(
            filters=ffn_channel, kernel_size=1, strides=1, activation="relu"
        )
        self.conv5 = keras.layers.Conv2D(
            filters=input_channels, kernel_size=1, strides=1
        )

        self.beta = tf.Variable(
            tf.ones((1, 1, 1, input_channels)), trainable=self.balanced_skip_connection
        )
        self.gamma = tf.Variable(
            tf.ones((1, 1, 1, input_channels)), trainable=self.balanced_skip_connection
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        # Block 1
        x = self.conv1(inputs)
        x = self.dconv2(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        # Residual connection
        x = inputs + self.beta * x

        # Block 2
        y = self.conv4(x)
        y = self.conv5(y)
        y = self.dropout2(y)

        # Residual connection
        y = x + self.gamma * y

        return y

    def get_config(self) -> dict:
        """Add constructor arguments to the config"""
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "factor": self.factor,
                "drop_out_rate": self.drop_out_rate,
                "balanced_skip_connection": self.balanced_skip_connection,
            }
        )
        return config
