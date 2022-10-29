import numpy as np
import tensorflow as tf


class DownBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, channel_factor: float, *args, **kwargs):
        super(DownBlock, self).__init__(*args, **kwargs)
        self.average_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv = tf.keras.layers.Conv2D(
            int(channels * channel_factor), kernel_size=1, strides=1, padding="same"
        )

    def call(self, inputs, *args, **kwargs):
        return self.conv(self.average_pool(inputs))


class DownSampleBlock(tf.keras.layers.Layer):
    def __init__(
        self, channels: int, scale_factor: int, channel_factor: float, *args, **kwargs
    ):
        super(DownSampleBlock, self).__init__(*args, **kwargs)
        layers = []
        for _ in range(int(np.log2(scale_factor))):
            layers.append(DownBlock(channels, channel_factor))
            channels = int(channels * channel_factor)
        self.layers = tf.keras.Sequential(layers)

    def call(self, inputs, *args, **kwargs):
        return self.layers(inputs)
