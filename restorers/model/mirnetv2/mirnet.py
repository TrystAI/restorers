from typing import Dict

import tensorflow as tf

from .mrb import MultiScaleResidualBlock


class RecursiveResidualGroup(tf.keras.layers.Layer):
    """Implementation of the Recursive Residual Group.

    The Recursive Residual Group forms the basic building block on MirNetV2.
    It progressively breaks down the input signal in order to simplify the overall
    learning process, and allows the construction of very deep networks.

    Reference:

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


class MirNetv2(tf.keras.Model):
    """Implementation of the MirNetv2 model.

    MirNetv2 is a fully convolutional architecture that learns enriched feature
    representations for image restoration and enhancement. It is based on a
    **recursive residual design** with the **multi-scale residual block** or **MRB**
    at its core. The main branch of the MRB is dedicated to maintaining spatially-precise
    high-resolution representations through the entire network and the complimentary set
    of parallel branches provide better contextualized features.

    Reference:

    1. [Learning Enriched Features for Fast Image Restoration and Enhancement](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/zamir-2022-mirnetv2.pdf)
    2. [Official PyTorch implementation of MirNetv2](https://github.com/swz30/MIRNetv2/blob/main/basicsr/models/archs/mirnet_v2_arch.py#L242)

    Args:
        channels (int): number of channels in the feature map.
        channel_factor (float): factor by which number of the number of output channels vary.
        num_mrb_blocks (int): number of multi-scale residual blocks.
        add_residual_connection (bool): add a residual connection between the inputs and the
            outputs or not.
    """

    def __init__(
        self,
        channels: int,
        channel_factor: float,
        num_mrb_blocks: int,
        add_residual_connection: bool,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.channel_factor = channel_factor
        self.num_mrb_blocks = num_mrb_blocks
        self.add_residual_connection = add_residual_connection

        self.conv_in = tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same")

        self.rrg_block_1 = RecursiveResidualGroup(
            channels, num_mrb_blocks, channel_factor, groups=1
        )
        self.rrg_block_2 = RecursiveResidualGroup(
            channels, num_mrb_blocks, channel_factor, groups=2
        )
        self.rrg_block_3 = RecursiveResidualGroup(
            channels, num_mrb_blocks, channel_factor, groups=4
        )
        self.rrg_block_4 = RecursiveResidualGroup(
            channels, num_mrb_blocks, channel_factor, groups=4
        )

        self.conv_out = tf.keras.layers.Conv2D(3, kernel_size=3, padding="same")

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        shallow_features = self.conv_in(inputs)
        deep_features = self.rrg_block_1(shallow_features)
        deep_features = self.rrg_block_2(deep_features)
        deep_features = self.rrg_block_3(deep_features)
        deep_features = self.rrg_block_4(deep_features)
        output = self.conv_out(deep_features)
        output = output + inputs if self.add_residual_connection else output
        return output

    def save(self, filepath: str, *args, **kwargs) -> None:
        input_tensor = tf.keras.Input(shape=[None, None, 3])
        saved_model = tf.keras.Model(
            inputs=input_tensor, outputs=self.call(input_tensor)
        )
        saved_model.save(filepath, *args, **kwargs)

    def get_config(self) -> Dict:
        return {
            "channels": self.channels,
            "num_mrb_blocks": self.num_mrb_blocks,
            "channel_factor": self.channel_factor,
            "add_residual_connection": self.add_residual_connection,
        }
