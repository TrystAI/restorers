from typing import Optional

import tensorflow as tf
from tensorflow import keras


class PlainBlock(keras.layers.Layer):
    """
    PlainBlock Layer

    This is the plain block proposed in NAFNet Paper.
    This is inspired from the restormer's block, keeping the most common components.

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

        self.activation = keras.layers.Activation("relu")

        self.dropout1 = keras.layers.Dropout(drop_out_rate)

        self.dropout2 = keras.layers.Dropout(drop_out_rate)

    def get_dw_channel(self, input_channels: int) -> int:
        return input_channels

    def get_ffn_channel(self, input_channels: int) -> int:
        return input_channels * self.factor

    def get_attention_layer(self, input_shape: tf.TensorShape) -> None:
        return keras.layers.Identity()

    def build(self, input_shape: tf.TensorShape) -> None:
        input_channels = input_shape[-1]
        dw_channel = self.get_dw_channel(input_channels)

        self.conv1 = keras.layers.Conv2D(filters=dw_channel, kernel_size=1, strides=1)
        self.dconv2 = keras.layers.Conv2D(
            filters=dw_channel,
            kernel_size=3,
            padding="same",
            strides=1,
            groups=dw_channel,
        )

        self.attention = self.get_attention_layer(input_shape)

        self.conv3 = keras.layers.Conv2D(
            filters=input_channels, kernel_size=1, strides=1
        )

        ffn_channel = self.get_ffn_channel(input_channels)

        self.conv4 = keras.layers.Conv2D(filters=ffn_channel, kernel_size=1, strides=1)
        self.conv5 = keras.layers.Conv2D(
            filters=input_channels, kernel_size=1, strides=1
        )

        self.beta = tf.Variable(
            tf.ones((1, 1, 1, input_channels)), trainable=self.balanced_skip_connection
        )
        self.gamma = tf.Variable(
            tf.ones((1, 1, 1, input_channels)), trainable=self.balanced_skip_connection
        )

    def call_block1(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.conv1(inputs)
        x = self.dconv2(x)
        x = self.activation(x)
        x = self.attention(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        return x

    def call_block2(self, inputs: tf.Tensor) -> tf.Tensor:
        y = self.conv4(inputs)
        y = self.activation(y)
        y = self.conv5(y)
        y = self.dropout2(y)
        return y

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        # Block 1
        x = self.call_block1(inputs)

        # Residual connection
        x = inputs + self.beta * x

        # Block 2
        y = self.call_block2(x)

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
