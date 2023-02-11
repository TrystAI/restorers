from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from .nafblock import NAFBlock


class PixelShuffle(keras.layers.Layer):
    """
    PixelShuffle Layer

    Wrapper Class for tf.nn.depth_to_space
    """

    def __init__(self, upscale_factor: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.upscale_factor = upscale_factor

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return tf.nn.depth_to_space(inputs, self.upscale_factor)


class NAFNet(keras.models.Model):
    """
    NAFNet

    The input channels will be mapped to the number of filters passed.
    After each down block, the number of filters will increase by a factor of 2.
    After each up block, the number of filters will decrease by a factor of 2.
    And finally the filters will be mapped back to the initial input size.

    Overwrite create_encoder_and_down_blocks, create_decoder_and_up_blocks, create_middle_blocks
    to add your own implementation for these blocks. But make sure to follow the restrictions on
    these methods.

    Parameters:
        filters: denotes the starting filter size.
        middle_block_num: denotes the number of middle blocks.
            Each middle block is a single NAFBlock unit.
        encoder_block_nums: (tuple) the tuple size denotes the number of encoder blocks.
            Each tuple entry denotes the number of NAFBlocks in the corresponding encoder block.
            len(encoder_block_nums) should be the same as the len(decoder_block_nums)
        decoder_block_nums: (tuple) the tuple size denotes the number of decoder blocks.
            Each tuple entry denotes the number of NAFBlocks in the corresponding decoder block.
            len(decoder_block_nums) should be the same as the len(encoder_block_nums)
    """

    def __init__(
        self,
        filters: Optional[int] = 16,
        middle_block_num: Optional[int] = 1,
        encoder_block_nums: Optional[Tuple[int]] = (1, 1, 1, 1),
        decoder_block_nums: Optional[Tuple[int]] = (1, 1, 1, 1),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.intro = keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same")
        self.ending = None

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

        self.create_middle_blocks(channels, middle_block_num)

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

    def build(self, input_shape: tf.TensorShape) -> None:
        input_channels = input_shape[-1]
        self.ending = keras.layers.Conv2DTranspose(
            filters=input_channels, kernel_size=3, padding="same"
        )

    def create_encoder_and_down_blocks(
        self,
        channels: int,
        encoder_block_nums: : Optional[Tuple[int]],
    ) -> int:
        """
        Creates equal number of encoder blocks and down blocks.
        """

        for num in encoder_block_nums:
            self.encoders.append(
                keras.models.Sequential([NAFBlock(channels) for _ in range(num)])
            )
            self.downs.append(
                keras.layers.Conv2D(2 * channels, kernel_size=2, strides=2)
            )
            channels *= 2
        return channels

    def create_middle_blocks(self, channels: int, middle_block_num: Optional[int]) -> None:
        self.middle_blocks = keras.models.Sequential(
            [NAFBlock(channels) for _ in range(middle_block_num)]
        )

    def create_decoder_and_up_blocks(
        self,
        channels: int,
        decoder_block_nums: Optional[Tuple[int]],
    ) -> int:
        """
        Creates equal number of decoder blocks and up blocks.
        """
        for num in decoder_block_nums:
            self.ups.append(
                keras.models.Sequential(
                    [
                        keras.layers.Conv2D(
                            2 * channels, kernel_size=1, strides=1, use_bias=False
                        ),
                        PixelShuffle(2),
                    ]
                )
            )
            channels = channels // 2
            self.decoders.append(
                keras.models.Sequential([NAFBlock(channels) for _ in range(num)])
            )
        return channels

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
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

        return x
