from typing import Optional, Dict

import tensorflow as tf

from .downsample import DownSampleBlock
from .rcb import ResidualContextBlock
from .skff import SelectiveKernelFeatureFusion
from .upsample import UpSampleBlock


class MultiScaleResidualBlock(tf.keras.layers.Layer):
    def __init__(
        self, channels: int, channel_factor: float, groups: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

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
