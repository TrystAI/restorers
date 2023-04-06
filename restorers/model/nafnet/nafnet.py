from typing import Tuple, Type

import tensorflow as tf
from tensorflow import keras

from .nafblock import NAFBlock
from .nafblock import PLAIN, BASELINE, NAFBLOCK
from .blocks import PixelShuffle, UpScale


class NAFNet(keras.models.Model):
    """
    NAFNet

    The input channels will be mapped to the number of filters passed.
    After each down block, the number of filters will increase by a factor of 2.
    After each up block, the number of filters will decrease by a factor of 2.
    And finally the filters will be mapped back to the initial input size.

    Overwrite `create_encoder_and_down_blocks`, `create_decoder_and_up_blocks`,
    and `create_middle_blocks` to add your own implementation for these blocks.
    Overwrite get_blocks to use your custom block in NAFNet. But make sure to follow
    the restrictions on these methods and blocks.

    ![](https://i.imgur.com/Ll017JJ.png)

    Reference:

    1. [Simple Baselines for Image Restoration](https://arxiv.org/abs/2204.04676)

    Args:
        filters (Optional[int]): denotes the starting filter size.
            Default filters' size is 16 .
        middle_block_num (Optional[int]): denotes the number of middle blocks.
            Each middle block is a single NAFBlock unit. Default value is 1.
        encoder_block_nums (tuple): the tuple size denotes the number of encoder blocks.
            Each tuple entry denotes the number of NAFBlocks in the corresponding encoder block.
            len(encoder_block_nums) should be the same as the len(decoder_block_nums)
            Default value is (1,1,1,1).
        decoder_block_nums (tuple): the tuple size denotes the number of decoder blocks.
            Each tuple entry denotes the number of NAFBlocks in the corresponding decoder block.
            len(decoder_block_nums) should be the same as the len(encoder_block_nums)
            Default value is (1,1,1,1).
        block_type (str): denotes what block to use in NAFNet
            Default block is 'nafblock'
    """

    def __init__(
        self,
        filters: int = 16,
        middle_block_num: int = 1,
        encoder_block_nums: Tuple[int] = (1, 1, 1, 1),
        decoder_block_nums: Tuple[int] = (1, 1, 1, 1),
        block_type: str = NAFBLOCK,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.filters = filters
        self.middle_block_num = middle_block_num
        self.encoder_block_nums = encoder_block_nums
        self.decoder_block_nums = decoder_block_nums
        self.block_type = block_type

        self.intro = keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same")

        self.encoders = []
        self.decoders = []
        self.ups = []
        self.downs = []

        if len(encoder_block_nums) != len(decoder_block_nums):
            raise ValueError(
                "The number of encoder blocks should match the number of decoder blocks"
                f"In the constructor {len(encoder_block_nums)} encoder blocks"
                f" and {len(decoder_block_nums)} were passed."
            )

        channels = filters
        channels = self.create_encoder_and_down_blocks(channels, encoder_block_nums)

        if len(self.encoders) != len(self.downs):
            raise ValueError(
                "The number of encoder blocks should match the number of down blocks"
                f"In `create_encoder_and_down_blocks` {len(self.encoders)} encoder blocks"
                f" and {len(self.downs)} down blocks were created."
            )

        self.create_middle_blocks(middle_block_num)

        self.create_decoder_and_up_blocks(channels, decoder_block_nums)

        if len(self.decoders) != len(self.ups):
            raise ValueError(
                "The number of decoder blocks should match the number of up blocks"
                f"In `create_decoder_and_up_blocks` {len(self.decoders)} decoder blocks"
                f" and {len(self.ups)} up blocks were created."
            )

        if len(encoder_block_nums) != len(decoder_block_nums):
            raise ValueError(
                "The number of encoder blocks should match the number of decoder blocks"
                f"In `create_encoder_and_down_blocks` {len(self.encoders)} encoder blocks were created."
                f"In `create_decoder_and_up_blocks` {len(self.decoders)} decoder blocks were created."
            )

        # The height and width of the image should be a
        #  multiple of self.expected_image_scale
        # If that is not the case, it will be fixed in the call(...) method.
        self.expected_image_scale = 2 ** len(self.encoders)

    def build(self, input_shape: tf.TensorShape) -> None:
        input_channels = input_shape[-1]
        self.ending = keras.layers.Conv2DTranspose(
            filters=input_channels, kernel_size=3, padding="same"
        )

    def get_block(self) -> keras.layers.Layer:
        """Returns the block to be used in NAFNet. This function can be overriden to use custom blocks
        in NAFNet.
        """
        return NAFBlock(mode=self.block_type)

    def create_encoder_and_down_blocks(
        self,
        channels: int,
        encoder_block_nums: Tuple[int],
    ) -> int:
        """Creates equal number of encoder blocks and down blocks."""
        for num in encoder_block_nums:
            self.encoders.append(
                keras.models.Sequential([self.get_block() for _ in range(num)])
            )
            self.downs.append(
                keras.layers.Conv2D(2 * channels, kernel_size=2, strides=2)
            )
            channels *= 2
        return channels

    def create_middle_blocks(self, middle_block_num: int) -> None:
        """Creates middle blocks in NAFNet"""
        self.middle_blocks = keras.models.Sequential(
            [self.get_block() for _ in range(middle_block_num)]
        )

    def create_decoder_and_up_blocks(
        self,
        channels: int,
        decoder_block_nums: Tuple[int],
    ) -> int:
        """Creates equal number of decoder blocks and up blocks."""
        for num in decoder_block_nums:
            self.ups.append(UpScale(2 * channels, pixel_shuffle_factor=2))
            channels = channels // 2
            self.decoders.append(
                keras.models.Sequential([self.get_block() for _ in range(num)])
            )
        return channels

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        _, H, W, _ = inputs.shape

        # Scale the image to the next nearest multiple of self.expected_image_scale
        inputs = self.fix_input_shape(inputs)

        x = self.intro(inputs)

        encoder_outputs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encoder_outputs.append(x)
            x = down(x)

        x = self.middle_blocks(x)

        for decoder, up, encoder_output in zip(
            self.decoders, self.ups, encoder_outputs[::-1]
        ):
            x = up(x)
            # Residual connection of encoder blocks with decoder blocks
            x = x + encoder_output
            x = decoder(x)

        x = self.ending(x)
        # Residual connection of inputs with output
        x = x + inputs

        # Crop back to the original size
        return x[:, :H, :W, :]

    def fix_input_shape(self, inputs: tf.Tensor) -> tf.Tensor:
        """Fixes input shape for NAFNet.
        This is because NAFNet can only work with images whose shape is multiple of
        2**(no. of encoder blocks). Hence the image is padded to match that shape.
        """
        _, H, W, _ = inputs.shape

        # Calculating how much padding is required
        height_padding, width_padding = 0, 0
        if H % self.expected_image_scale != 0:
            height_padding = self.expected_image_scale - H % self.expected_image_scale
        if W % self.expected_image_scale != 0:
            width_padding = self.expected_image_scale - W % self.expected_image_scale

        paddings = tf.constant(
            [[0, 0], [0, height_padding], [0, width_padding], [0, 0]]
        )
        return tf.pad(inputs, paddings)

    def save(self, filepath: str, *args, **kwargs) -> None:
        input_tensor = tf.keras.Input(shape=[None, None, 3])
        saved_model = tf.keras.Model(
            inputs=input_tensor, outputs=self.call(input_tensor)
        )
        saved_model.save(filepath, *args, **kwargs)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "middle_block_num": self.middle_block_num,
                "encoder_block_nums": self.encoder_block_nums,
                "decoder_block_nums": self.decoder_block_nums,
                "block_type": self.block_type,
            }
        )
        return config
