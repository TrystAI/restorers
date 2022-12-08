import numpy as np
import tensorflow as tf


class DownBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, channel_factor: float, *args, **kwargs):
        super(DownBlock, self).__init__(*args, **kwargs)
        self.channels = channels
        self.channel_factor = channel_factor
        self.average_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv = tf.keras.layers.Conv2D(
            int(self.channels * self.channel_factor),
            kernel_size=1,
            strides=1,
            padding="same",
        )

    def call(self, inputs, *args, **kwargs):
        return self.conv(self.average_pool(inputs))

    def get_config(self):
        return {"channels": self.channels, "channel_factor": self.channel_factor}


class DownSampleBlock(tf.keras.layers.Layer):
    def __init__(
        self, channels: int, scale_factor: int, channel_factor: float, *args, **kwargs
    ):
        super(DownSampleBlock, self).__init__(*args, **kwargs)
        self.channels = channels
        self.scale_factor = scale_factor
        self.channel_factor = channel_factor
        self.layers = []
        for _ in range(int(np.log2(self.scale_factor))):
            self.layers.append(DownBlock(channels, self.channel_factor))
            channels = int(channels * self.channel_factor)

    def call(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_config(self):
        return {
            "channels": self.channels,
            "scale_factor": self.scale_factor,
            "channel_factor": self.channel_factor,
        }
