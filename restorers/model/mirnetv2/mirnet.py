from typing import Dict

import tensorflow as tf

from .recursive_residual_group import RecursiveResidualGroup


class MirNetv2(tf.keras.Model):
    """Implementation of the MirNetv2 model.

    MirNetv2 is a fully convolutional architecture that learns enriched feature
    representations for image restoration and enhancement. It is based on a
    **recursive residual design** with the **multi-scale residual block** or **MRB**
    at its core. The main branch of the MRB is dedicated to maintaining spatially-precise
    high-resolution representations through the entire network and the complimentary set
    of parallel branches provide better contextualized features.

    ![The MirNetv2 Architecture](https://i.imgur.com/oCIo69j.png){ loading=lazy }

    ??? example "Examples"
        - [Training a supervised low-light enhancement model using MirNetv2](../../examples/train_mirnetv2).

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
