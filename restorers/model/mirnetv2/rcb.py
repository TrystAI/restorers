import tensorflow as tf

from transformers.tf_utils import shape_list


class ContextBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channels = channels

        self.mask_conv = tf.keras.layers.Conv2D(1, kernel_size=1, padding="same")

        self.channel_add_conv_1 = tf.keras.layers.Conv2D(
            self.channels, kernel_size=1, padding="same"
        )
        self.channel_add_conv_2 = tf.keras.layers.Conv2D(
            self.channels, kernel_size=1, padding="same"
        )

        self.softmax = tf.keras.layers.Softmax(axis=1, dtype="float32")

    def modeling(self, inputs):
        _, height, width, channels = shape_list(inputs)
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

    def call(self, inputs, *args, **kwargs):
        context = self.modeling(inputs)
        channel_add_term = self.channel_add_conv_1(context)
        channel_add_term = tf.nn.leaky_relu(channel_add_term, alpha=0.2)
        channel_add_term = self.channel_add_conv_2(channel_add_term)
        return inputs + channel_add_term

    def get_config(self):
        return {"channels": self.channels}


class ResidualContextBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, groups: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.groups = groups

        self.conv_1 = tf.keras.layers.Conv2D(
            self.channels, kernel_size=3, padding="same", groups=self.groups
        )
        self.conv_2 = tf.keras.layers.Conv2D(
            self.channels, kernel_size=3, padding="same", groups=self.groups
        )

        self.context_block = ContextBlock(channels=self.channels)

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.conv_2(x)
        x = self.context_block(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = x + inputs
        return x

    def get_config(self):
        return {"channels": self.channels, "groups": self.groups}
