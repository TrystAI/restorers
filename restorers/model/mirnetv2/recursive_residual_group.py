from typing import Dict

import tensorflow as tf

from .mrb import MultiScaleResidualBlock


class RecursiveResidualGroup(tf.keras.layers.Layer):
    """Implementation of the Recursive Residual Group.

    The Recursive Residual Group forms the basic building block on MirNetV2.
    It progressively breaks down the input signal in order to simplify the overall
    learning process, and allows the construction of very deep networks.

    !!! info "References"
        1. [Learning Enriched Features for Fast Image Restoration and Enhancement](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/zamir-2022-mirnetv2.pdf)
        2. [Official PyTorch implementation of MirNetv2](https://github.com/swz30/MIRNetv2/blob/main/basicsr/models/archs/mirnet_v2_arch.py#L242)

    Args:
        channels (int): number of channels in the feature map.
        num_mrb_blocks (int): number of multi-scale residual blocks.
        channel_factor (float): factor by which number of the number of output channels vary.
        groups (int): number of groups in which the input is split along the
            channel axis in the convolution layers.
    """

    def __init__(
        self,
        channels: int,
        num_mrb_blocks: int,
        channel_factor: float,
        groups: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.num_mrb_blocks = num_mrb_blocks
        self.channel_factor = channel_factor
        self.groups = groups

        self.layers = [
            MultiScaleResidualBlock(channels, channel_factor, groups)
            for _ in range(num_mrb_blocks)
        ]
        self.layers.append(
            tf.keras.layers.Conv2D(channels, kernel_size=3, strides=1, padding="same")
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        residual = inputs
        for layer in self.layers:
            residual = layer(residual)
        residual = residual + inputs
        return residual

    def get_config(self) -> Dict:
        return {
            "channels": self.channels,
            "num_mrb_blocks": self.num_mrb_blocks,
            "channel_factor": self.channel_factor,
            "groups": self.groups,
        }
