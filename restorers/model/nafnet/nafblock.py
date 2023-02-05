import tensorflow as tf
from tensorflow import keras


class SimpleGate(keras.layers.Layer):
    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor

    def call(self, x):
        x = tf.expand_dims(x, axis=-1)
        return tf.reduce_prod(
            tf.concat(tf.split(x, num_or_size_splits=self.factor, axis=-2), axis=-1),
            axis=-1,
        )


class SimplifiedChannelAttention(keras.layers.Layer):
    def __init__(self, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.conv = keras.layers.Conv2D(filters=channels, kernel_size=1)

    def call(self, inputs):
        average_pooling = self.avg_pool(inputs)
        feature_descriptor = tf.reshape(
            average_pooling, shape=(-1, 1, 1, self.channels)
        )
        features = self.conv(feature_descriptor)
        return inputs * features


class NAFBlock(keras.layers.Layer):
    def __init__(
        self,
        input_channels,
        factor=2,
        drop_out_rate=0.0,
        balanced_skip_connection=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.factor = factor
        dw_channel = input_channels * factor

        self.layer_norm1 = keras.layers.LayerNormalization()

        self.conv1 = keras.layers.Conv2D(filters=dw_channel, kernel_size=1, strides=1)
        self.dconv2 = keras.layers.Conv2D(
            filters=dw_channel,
            kernel_size=1,
            padding="same",
            strides=1,
            groups=dw_channel,
        )

        self.simple_gate = SimpleGate(factor)
        self.simplified_attention = SimplifiedChannelAttention(input_channels)

        self.conv3 = keras.layers.Conv2D(
            filters=input_channels, kernel_size=1, strides=1
        )

        self.dropout1 = keras.layers.Dropout(drop_out_rate)

        self.layer_norm2 = keras.layers.LayerNormalization()

        ffn_channel = input_channels * factor

        self.conv4 = keras.layers.Conv2D(filters=ffn_channel, kernel_size=1, strides=1)
        self.conv5 = keras.layers.Conv2D(
            filters=input_channels, kernel_size=1, strides=1
        )

        self.dropout2 = keras.layers.Dropout(drop_out_rate)

        self.beta = tf.Variable(
            tf.ones((1, 1, 1, input_channels)), trainable=balanced_skip_connection
        )
        self.gamma = tf.Variable(
            tf.ones((1, 1, 1, input_channels)), trainable=balanced_skip_connection
        )

    def call(self, inputs):

        # Block 1
        x = self.layer_norm1(inputs)
        x = self.conv1(x)
        x = self.dconv2(x)
        x = self.simple_gate(x)
        x = self.simplified_attention(x)
        x = self.conv3(x)

        # Residual connection
        x = x + self.beta * inputs

        # Block 2
        y = self.layer_norm2(x)
        y = self.conv4(y)
        y = self.simple_gate(y)
        y = self.conv5(y)

        print(y.shape, x.shape)
        # Residual connection
        y = y + self.gamma * x

        return y
