from typing import Optional

import tensorflow as tf
from tensorflow import keras

NAFBLOCK = "nafblock"
PLAIN = "plain"
BASELINE = "baseline"


class SimpleGate(keras.layers.Layer):
    """
    Simple Gate
    It splits the input of size (b,h,w,c) into tensors of size (b,h,w,c//factor) and returns their Hadamard product

    Reference: NAFNet Paper (Simple Baselines for Image Restoration)
    https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670017.pdf

    Parameters:
        factor: the amount by which the channels are scaled down
    """

    def __init__(self, factor: Optional[int] = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        x = tf.expand_dims(x, axis=-1)
        return tf.reduce_prod(
            tf.concat(tf.split(x, num_or_size_splits=self.factor, axis=-2), axis=-1),
            axis=-1,
        )

    def get_config(self) -> dict:
        """Add factor to the config"""
        config = super().get_config()
        config.update({"factor": self.factor})
        return config


class ChannelAttention(keras.layers.Layer):
    """
    Channel Attention layer

    The block is named Squeeze-and-Excitation block (SE Block) in the original paper.
    First the input is 'squeezed' across the spatial dimension to generate
        a channel-wise descriptor.
    Following that the inter channel dependency is learnt by applying
        two convolution layers.
    Then finally, the input is rescaled by a channel-wise multiplication with the
        output of the excitation operation.

    Reference: Squeeze-and-Excitation Networks, Hu et al.
    https://ieeexplore.ieee.org/document/8578843

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


class SimplifiedChannelAttention(keras.layers.Layer):
    """
    Simplified Channel Attention layer
    It is a modification of channel attention without any non-linear activations.

    The Squeeze and final rescaling step is identical to the ChannelAttention Layer.
    But following the philosophy of NAFNet paper, the excitation operation with
        two conv layers with respective activations are replaced with a single conv
        block. So the inter channel dependency is learnt but any gate or activation
        is not used.
        (Check the paper/doc string of NAFBlock for more details)

    Reference: NAFNet Paper (Simple Baselines for Image Restoration)
    https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670017.pdf

    Parameters:
        channels: number of channels in input
    """

    def __init__(self, channels: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.conv = keras.layers.Conv2D(filters=channels, kernel_size=1)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        average_pooling = self.avg_pool(inputs)
        feature_descriptor = tf.reshape(
            average_pooling, shape=(-1, 1, 1, self.channels)
        )
        features = self.conv(feature_descriptor)
        return inputs * features

    def get_config(self) -> dict:
        """Add channels to the config"""
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


class NAFBlock(keras.layers.Layer):
    """
    NAFBlock (Nonlinear Activation Free Block)

    The authors first define a plain block by retaining the most used operations
        from the restormer block.
    In the plain block layer normalization and channel attention is added to make
        the baseline block.
    NAFBlock is constructed by removing all the non-linear activations from
        the baseline block.

    The authors have the idea that any operations of the form,
    .. math::
        f(X) \dot \sigma(g(Y))
    (where f and g are feature maps and \sigma is activation function)
    can be simplified to the form
    .. math::
        X \dot g(Y)
    Using this idea, all the nonlinear activations are replaced by
        a series of Hadamard produces

    Reference: NAFNet Paper (Simple Baselines for Image Restoration)
    https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670017.pdf

    Parameters:
        input_channels: number of channels in the input (as NAFBlock retains the input size in the output)
        factor: factor by which the channels must be increased before being reduced by simple gate.
            (Higher factor denotes higher order polynomial in multiplication. Default factor is 2)
        drop_out_rate: dropout rate
        balanced_skip_connection: adds additional trainable parameters to the skip connections.
            The parameter denotes how much importance should be given to the sub block in the skip connection.
        mode: NAFBlock has 3 mode.
            'plain' mode uses the PlainBlock.
                It is derived from the restormer block, keeping the most common components
            'baseline' mode used the BaselineBlock
                It is derived by adding layer normalization, channel attention to PlainBlock.
                It also replaces ReLU activation with GeLU in PlainBlock.
            'nafblock' mode uses the NAFBlock
                It derived from BaselineBlock by removing all the non-linear activation.
                Non-linear activations are replaced by equivalent matrix multiplication operations.
    """

    def __init__(
        self,
        factor: Optional[int] = 2,
        drop_out_rate: Optional[float] = 0.0,
        balanced_skip_connection: Optional[bool] = False,
        mode: Optional[str] = NAFBLOCK,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.factor = factor
        self.drop_out_rate = drop_out_rate
        self.balanced_skip_connection = balanced_skip_connection

        valid_mode = {PLAIN, BASELINE, NAFBLOCK}
        if mode not in valid_mode:
            raise ValueError("Mode must be one of %r." % valid_mode)
        self.mode = mode

        if self.mode == PLAIN:
            self.activation = keras.layers.Activation("relu")
        elif self.mode == BASELINE:
            self.activation = keras.layers.Activation("gelu")
        else:
            self.activation = SimpleGate(factor)

        self.dropout1 = keras.layers.Dropout(drop_out_rate)

        self.dropout2 = keras.layers.Dropout(drop_out_rate)

        self.layer_norm1 = None
        self.layer_norm2 = None
        if self.mode in [NAFBLOCK, BASELINE]:
            self.layer_norm1 = keras.layers.LayerNormalization()
            self.layer_norm2 = keras.layers.LayerNormalization()

    def get_dw_channel(self, input_channels: int) -> int:
        if self.mode == NAFBLOCK:
            return input_channels * self.factor
        else:
            return input_channels

    def get_ffn_channel(self, input_channels: int) -> int:
        return input_channels * self.factor

    def get_attention_layer(
        self, input_shape: tf.TensorShape
    ) -> Optional[keras.layers.Layer]:
        input_channels = input_shape[-1]
        if self.mode == NAFBLOCK:
            return SimplifiedChannelAttention(input_channels)
        elif self.mode == BASELINE:
            return ChannelAttention(input_channels)
        else:
            return None

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
        x = inputs
        if self.layer_norm1 != None:
            x = self.layer_norm1(x)
        x = self.conv1(x)
        x = self.dconv2(x)
        x = self.activation(x)
        if self.attention != None:
            x = self.attention(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        return x

    def call_block2(self, inputs: tf.Tensor) -> tf.Tensor:
        y = inputs
        if self.layer_norm2 != None:
            y = self.layer_norm2(y)
        y = self.conv4(y)
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
                "factor": self.factor,
                "drop_out_rate": self.drop_out_rate,
                "balanced_skip_connection": self.balanced_skip_connection,
                "mode": self.mode,
            }
        )
        return config
