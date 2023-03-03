from typing import Dict

import tensorflow as tf


class ContextBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.channels = channels

        self.mask_conv = tf.keras.layers.Conv2D(1, kernel_size=1, padding="same")

        self.channel_add_conv_1 = tf.keras.layers.Conv2D(
            channels, kernel_size=1, padding="same"
        )
        self.channel_add_conv_2 = tf.keras.layers.Conv2D(
            channels, kernel_size=1, padding="same"
        )

        self.softmax = tf.keras.layers.Softmax(axis=1)
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def modeling(self, inputs: tf.Tensor) -> tf.Tensor:
        _, height, width, channels = [
            tf.shape(inputs)[_shape_idx] if _shape is None else _shape
            for _shape_idx, _shape in enumerate(inputs.shape.as_list())
        ]
        reshaped_inputs = tf.expand_dims(
            tf.reshape(inputs, (-1, channels, height * width)), axis=1
        )

        context_mask = self.mask_conv(inputs)
        context_mask = tf.reshape(context_mask, (-1, height * width, 1))
        context_mask = self.softmax(context_mask)
        context_mask = tf.expand_dims(context_mask, axis=1)

        context = tf.reshape(
            tf.matmul(reshaped_inputs, context_mask), (-1, 1, 1, channels)
        )
        return context

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        context = self.modeling(inputs)
        channel_add_term = self.channel_add_conv_1(context)
        channel_add_term = self.leaky_relu(channel_add_term)
        channel_add_term = self.channel_add_conv_2(channel_add_term)
        return inputs + channel_add_term

    def get_config(self) -> Dict:
        return {"channels": self.channels}


class ResidualContextBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, groups: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.groups = groups

        self.conv_1 = tf.keras.layers.Conv2D(
            channels, kernel_size=3, padding="same", groups=groups
        )
        self.conv_2 = tf.keras.layers.Conv2D(
            channels, kernel_size=3, padding="same", groups=groups
        )
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.context_block = ContextBlock(channels=channels)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.conv_1(inputs)
        x = self.leaky_relu(x)
        x = self.conv_2(x)
        x = self.context_block(x)
        x = self.leaky_relu(x)
        x = x + inputs
        return x

    def get_config(self) -> Dict:
        return {"channels": self.channels, "groups": self.groups}
