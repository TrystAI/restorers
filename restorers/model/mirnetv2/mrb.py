from typing import Optional, Dict

import tensorflow as tf

from .downsample import DownSampleBlock
from .rcb import ResidualContextBlock
from .skff import SelectiveKernelFeatureFusion
from .upsample import UpSampleBlock


class MultiScaleResidualBlock(tf.keras.layers.Layer):
    """Implementation of the Multi-scale Residual Block.

    The Multi-scale Residual Block mechanism of collecting multiscale spatial information.
    This block forms the core component of the recursive residual design of MIRNet-v2.
    The key advantages of MRB are:

    - It is capable of generating a spatially-precise output by maintaining high-resolution representations, while receiving rich contextual information from low-resolutions.

    - It allows contextualized-information transfer from the low-resolution streams to consolidate the high-resolution features.

    Reference:

    1. [Learning Enriched Features for Fast Image Restoration and Enhancement](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/zamir-2022-mirnetv2.pdf)
    2. [Official PyTorch implementation of MirNetv2](https://github.com/swz30/MIRNetv2/blob/main/basicsr/models/archs/mirnet_v2_arch.py#L189)

    Args:
        channels (int): number of channels in the feature map.
        channel_factor (float): factor by which number of the number of output channels vary.
        groups (int): number of groups in which the input is split along the
            channel axis in the convolution layers.
    """

    def __init__(
        self, channels: int, channel_factor: float, groups: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.channel_factor = channel_factor
        self.groups = groups

        # Residual Context Blocks
        self.rcb_top = ResidualContextBlock(
            int(channels * channel_factor**0), groups=groups
        )
        self.rcb_middle = ResidualContextBlock(
            int(channels * channel_factor**1), groups=groups
        )
        self.rcb_bottom = ResidualContextBlock(
            int(channels * channel_factor**2), groups=groups
        )

        # Downsample Blocks
        self.down_2 = DownSampleBlock(
            channels=int((channel_factor**0) * channels),
            scale_factor=2,
            channel_factor=channel_factor,
        )
        self.down_4_1 = DownSampleBlock(
            channels=int((channel_factor**0) * channels),
            scale_factor=2,
            channel_factor=channel_factor,
        )
        self.down_4_2 = DownSampleBlock(
            channels=int((channel_factor**1) * channels),
            scale_factor=2,
            channel_factor=channel_factor,
        )

        # UpSample Blocks
        self.up21_1 = UpSampleBlock(
            channels=int((channel_factor**1) * channels),
            scale_factor=2,
            channel_factor=channel_factor,
        )
        self.up21_2 = UpSampleBlock(
            channels=int((channel_factor**1) * channels),
            scale_factor=2,
            channel_factor=channel_factor,
        )
        self.up32_1 = UpSampleBlock(
            channels=int((channel_factor**2) * channels),
            scale_factor=2,
            channel_factor=channel_factor,
        )
        self.up32_2 = UpSampleBlock(
            channels=int((channel_factor**2) * channels),
            scale_factor=2,
            channel_factor=channel_factor,
        )

        # SKFF Blocks
        self.skff_top = SelectiveKernelFeatureFusion(
            channels=int(channels * channel_factor**0)
        )
        self.skff_middle = SelectiveKernelFeatureFusion(
            channels=int(channels * channel_factor**1)
        )

        # Convolution
        self.conv_out = tf.keras.layers.Conv2D(channels, kernel_size=1, padding="same")

    def call(self, inputs, *args, **kwargs):
        x_top = inputs
        x_middle = self.down_2(x_top)
        x_bottom = self.down_4_2(self.down_4_1(x_top))

        x_top = self.rcb_top(x_top)
        x_middle = self.rcb_middle(x_middle)
        x_bottom = self.rcb_bottom(x_bottom)

        x_middle = self.skff_middle([x_middle, self.up32_1(x_bottom)])
        x_top = self.skff_top([x_top, self.up21_1(x_middle)])

        x_top = self.rcb_top(x_top)
        x_middle = self.rcb_middle(x_middle)
        x_bottom = self.rcb_bottom(x_bottom)

        x_middle = self.skff_middle([x_middle, self.up32_2(x_bottom)])
        x_top = self.skff_top([x_top, self.up21_2(x_middle)])

        output = self.conv_out(x_top)
        output = output + inputs

        return output

    def get_config(self) -> Dict:
        return {
            "channels": self.channels,
            "channel_factor": self.channel_factor,
            "groups": self.groups,
        }
