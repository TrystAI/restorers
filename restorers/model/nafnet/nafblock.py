import tensorflow as tf
from tensorflow import keras


class SimpleGate(keras.layers.Layer):
    """
    Simple Gate
    It splits the input of size (b,h,w,c) into tensors of size (b,h,w,c//factor) and returns their Hadamard product
    Parameters:
        factor: the amount by which the channels are scaled down
    """

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
    """
    Simplified Channel Attention layer
    It is a modification of channel attention without any non-linear activations.
    Parameters:
        channels: number of channels in input
    """

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
    """
    NAFBlock (Nonlinear Activation Free Block)
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
        factor=2,
        drop_out_rate=0.0,
        balanced_skip_connection=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.factor = factor
        self.balanced_skip_connection = balanced_skip_connection

        self.layer_norm1 = keras.layers.LayerNormalization()

        self.conv1 = None
        self.dconv2 = None

        self.simple_gate = SimpleGate(factor)
        self.simplified_attention = None

        self.conv3 = None

        self.dropout1 = keras.layers.Dropout(drop_out_rate)

        self.layer_norm2 = keras.layers.LayerNormalization()

        self.conv4 = None
        self.conv5 = None

        self.dropout2 = keras.layers.Dropout(drop_out_rate)

        self.beta = None
        self.gamma = None

    def build(self, input_shape):
        input_channels = input_shape[-1]
        dw_channel = input_channels * self.factor

        self.conv1 = keras.layers.Conv2D(filters=dw_channel, kernel_size=1, strides=1)
        self.dconv2 = keras.layers.Conv2D(
            filters=dw_channel,
            kernel_size=1,
            padding="same",
            strides=1,
            groups=dw_channel,
        )

        self.simplified_attention = SimplifiedChannelAttention(input_channels)

        self.conv3 = keras.layers.Conv2D(
            filters=input_channels, kernel_size=1, strides=1
        )

        ffn_channel = input_channels * self.factor

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

    def call(self, inputs):
        # Block 1
        x = self.layer_norm1(inputs)
        x = self.conv1(x)
        x = self.dconv2(x)
        x = self.simple_gate(x)
        x = self.simplified_attention(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        # Residual connection
        x = inputs + self.beta * x

        # Block 2
        y = self.layer_norm2(x)
        y = self.conv4(y)
        y = self.simple_gate(y)
        y = self.conv5(y)
        y = self.dropout2(y)

        # Residual connection
        y = x + self.gamma * y

        return y
