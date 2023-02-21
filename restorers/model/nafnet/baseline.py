from typing import Optional

import tensorflow as tf
from tensorflow import keras


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


class BaselineBlock(keras.layers.Layer):
    """
    BaselineBlock Layer

    This is the baseline block proposed in NAFNet Paper.
    From this block all the non-linear activations are removed to generate the NAFBlock

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

        self.layer_norm1 = keras.layers.LayerNormalization()
        self.layer_norm2 = keras.layers.LayerNormalization()

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
            activation="gelu",
        )

        self.conv3 = keras.layers.Conv2D(
            filters=input_channels, kernel_size=1, strides=1
        )

        self.channel_attention = ChannelAttention(input_channels)

        ffn_channel = input_channels * self.factor

        self.conv4 = keras.layers.Conv2D(
            filters=ffn_channel, kernel_size=1, strides=1, activation="gelu"
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
        x = self.layer_norm1(inputs)
        x = self.conv1(x)
        x = self.dconv2(x)
        x = self.channel_attention(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        # Residual connection
        x = inputs + self.beta * x

        # Block 2
        y = self.layer_norm2(x)
        y = self.conv4(y)
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
