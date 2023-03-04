from typing import Dict

import numpy as np
import tensorflow as tf


class DownBlock(tf.keras.layers.Layer):
    """Submodule of `DownSampleBlock`.

    Reference:

    1. [Learning Enriched Features for Fast Image Restoration and Enhancement](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/zamir-2022-mirnetv2.pdf)
    2. [Official PyTorch implementation of MirNetv2](https://github.com/swz30/MIRNetv2/blob/main/basicsr/models/archs/mirnet_v2_arch.py#L130)

    Args:
        channels (int): number of input channels.
        channel_factor (float): factor by which number of the number of output channels vary.
    """

    def __init__(self, channels: int, channel_factor: float, *args, **kwargs) -> None:
        super(DownBlock, self).__init__(*args, **kwargs)

        self.channels = channels
        self.channel_factor = channel_factor

        self.average_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv = tf.keras.layers.Conv2D(
            int(channels * channel_factor), kernel_size=1, strides=1, padding="same"
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return self.conv(self.average_pool(inputs))

    def get_config(self) -> Dict:
        return {"channels": self.channels, "channel_factor": self.channel_factor}


class DownSampleBlock(tf.keras.layers.Layer):
    """Layer for downsampling feature map for the Multi-scale Residual Block.

    Reference:

    1. [Learning Enriched Features for Fast Image Restoration and Enhancement](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/zamir-2022-mirnetv2.pdf)
    2. [Official PyTorch implementation of MirNetv2](https://github.com/swz30/MIRNetv2/blob/main/basicsr/models/archs/mirnet_v2_arch.py#L142)

    Args:
        channels (int): number of input channels.
        scale_factor (int): number of downsample operations.
        channel_factor (float): factor by which number of the number of output channels vary.
    """

    def __init__(
        self, channels: int, scale_factor: int, channel_factor: float, *args, **kwargs
    ) -> None:
        super(DownSampleBlock, self).__init__(*args, **kwargs)

        self.channels = channels
        self.scale_factor = scale_factor
        self.channel_factor = channel_factor

        self.layers = []
        for _ in range(int(np.log2(scale_factor))):
            self.layers.append(DownBlock(channels, channel_factor))
            channels = int(channels * channel_factor)

    def call(self, x: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def get_config(self) -> Dict:
        return {
            "channels": self.channels,
            "channel_factor": self.channel_factor,
            "scale_factor": self.scale_factor,
        }
