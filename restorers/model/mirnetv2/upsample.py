from typing import Optional, Dict

import numpy as np
import tensorflow as tf


class UpBlock(tf.keras.layers.Layer):
    """Submodule of `UpSampleBlock`.

    Reference:

    1. [Learning Enriched Features for Fast Image Restoration and Enhancement](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/zamir-2022-mirnetv2.pdf)
    2. [Official PyTorch implementation of MirNetv2](https://github.com/swz30/MIRNetv2/blob/main/basicsr/models/archs/mirnet_v2_arch.py#L158)

    Args:
        channels (int): number of input channels.
        channel_factor (float): factor by which number of the number of output channels vary.
    """

    def __init__(self, channels: int, channel_factor: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = tf.keras.layers.Conv2D(
            int(channels // channel_factor), kernel_size=1, strides=1, padding="same"
        )
        self.upsample = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")

    def call(self, inputs, *args, **kwargs):
        return self.upsample(self.conv(inputs))


class UpSampleBlock(tf.keras.layers.Layer):
    """Layer for upsampling feature map for the Multi-scale Residual Block.

    Reference:

    1. [Learning Enriched Features for Fast Image Restoration and Enhancement](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/zamir-2022-mirnetv2.pdf)
    2. [Official PyTorch implementation of MirNetv2](https://github.com/swz30/MIRNetv2/blob/main/basicsr/models/archs/mirnet_v2_arch.py#L170)

    Args:
        channels (int): number of input channels.
        scale_factor (int): number of downsample operations.
        channel_factor (float): factor by which number of the number of output channels vary.
    """

    def __init__(
        self, channels: int, scale_factor: int, channel_factor: float, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.layers = []
        for _ in range(int(np.log2(scale_factor))):
            self.layers.append(UpBlock(channels, channel_factor))
            channels = int(channels // channel_factor)

    def call(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x
