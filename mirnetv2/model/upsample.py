import numpy as np
import tensorflow as tf


class UpBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, channel_factor: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.channel_factor = channel_factor

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            int(self.channels // self.channel_factor),
            kernel_size=1,
            strides=1,
            padding="same",
        )
        self.upsample = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")

    def call(self, inputs, *args, **kwargs):
        return self.upsample(self.conv(inputs))

    def get_config(self):
        return {"channels": self.channels, "channel_factor": self.channel_factor}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class UpSampleBlock(tf.keras.layers.Layer):
    def __init__(
        self, channels: int, scale_factor: int, channel_factor: float, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.scale_factor = scale_factor
        self.channel_factor = channel_factor

    def build(self, input_shape):
        self.layers = []
        channels = self.channels
        for _ in range(int(np.log2(self.scale_factor))):
            self.layers.append(UpBlock(channels, self.channel_factor))
            channels = int(channels // self.channel_factor)

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
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
