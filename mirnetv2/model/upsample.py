import numpy as np
import tensorflow as tf


class UpBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, channel_factor: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = tf.keras.layers.Conv2D(
            int(channels // channel_factor), kernel_size=1, strides=1, padding="same"
        )
        self.upsample = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")

    def call(self, inputs, *args, **kwargs):
        return self.upsample(self.conv(inputs))


class UpSampleBlock(tf.keras.layers.Layer):
    def __init__(
        self, channels: int, scale_factor: int, channel_factor: float, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        layers = []
        for _ in range(int(np.log2(scale_factor))):
            layers.append(UpBlock(channels, channel_factor))
            channels = int(channels // channel_factor)
        self.layers = tf.keras.Sequential(layers)

    def call(self, inputs, *args, **kwargs):
        return self.layers(inputs)
