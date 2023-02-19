from typing import Optional, Tuple, Type

import tensorflow as tf
from tensorflow import keras

from .nafblock import NAFBlock


class PixelShuffle(keras.layers.Layer):
    """
    PixelShuffle Layer

    Given input of size (H,W,C), it will generate an output
    of size
    (
        H*pixel_shuffle_factor,
        W*pixel_shuffle_factor,
        channels//(pixel_shuffle_factor**2)
    )

    Wrapper Class for tf.nn.depth_to_space
    """

    def __init__(self, upscale_factor: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.upscale_factor = upscale_factor

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return tf.nn.depth_to_space(inputs, self.upscale_factor)

    def get_config(self) -> dict:
        """Add upscale factor to the config"""
        config = super().get_config()
        config.update({"upscale_factor": self.upscale_factor})
        return config


class BlockStack(keras.layers.Layer):
    """
    BlockStack Layer

    Simple utility class to generate a sequential list of same layer
    """

    def __init__(
        self,
        block_class: Type[keras.layers.Layer],
        num_blocks: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.block_class = block_class
        self.num_blocks = num_blocks
        self.block_list = []
        self.args = args
        self.kwargs = kwargs

        for i in range(self.num_blocks):
            self.block_list.append(self.block_class(*args, **kwargs))

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        x = inputs
        for block in self.block_list:
            x = block(x)
        return x

    def get_config(self) -> dict:
        "Get config for BlockStack"
        config = super().get_config()
        config.update(
            {
                "block_class": self.block_class,
                "num_blocks": self.num_blocks,
                "args": self.args,
                "kwargs": self.kwargs,
            }
        )
        return config


class UpScale(keras.layers.Layer):
    """
    UpScale Layer

    Given channels and pixel_shuffle_factor as input, it will generate an output
    of size
    (
        H*pixel_shuffle_factor,
        W*pixel_shuffle_factor,
        channels//(pixel_shuffle_factor**2)
    )
    While giving input, make sure that (pixel_shuffle_factor**2) divides channels
    """

    def __init__(self, channels: int, pixel_shuffle_factor: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.pixel_shuffle_factor = pixel_shuffle_factor

        if channels % (pixel_shuffle_factor**2) != 0:
            raise ValueError(
                f"Number of channels must divide square of pixel_shuffle_factor"
                f"In the constructor {channels} channels and "
                f"{pixel_shuffle_factor} pixel_shuffle_factor was passed"
            )

        self.conv = keras.layers.Conv2D(
            channels, kernel_size=1, strides=1, use_bias=False
        )
        self.pixel_shuffle = PixelShuffle(pixel_shuffle_factor)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return self.pixel_shuffle(self.conv(inputs))

    def get_config(self) -> dict:
        """Add channels and pixel_shuffle_factor to the config"""
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "pixel_shuffle_factor": self.pixel_shuffle_factor,
            }
        )
        return config


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

        self.filters = filters
        self.middle_block_num = middle_block_num
        self.encoder_block_nums = encoder_block_nums
        self.decoder_block_nums = decoder_block_nums

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

        self.middle_blocks = BlockStack(NAFBlock, middle_block_num)

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
        encoder_block_nums: Optional[Tuple[int]],
    ) -> int:
        """
        Creates equal number of encoder blocks and down blocks.
        """

        for num in encoder_block_nums:
            self.encoders.append(BlockStack(NAFBlock, num))
            self.downs.append(
                keras.layers.Conv2D(2 * channels, kernel_size=2, strides=2)
            )
            channels *= 2
        return channels

    def create_decoder_and_up_blocks(
        self,
        channels: int,
        decoder_block_nums: Optional[Tuple[int]],
    ) -> int:
        """
        Creates equal number of decoder blocks and up blocks.
        """
        for num in decoder_block_nums:
            self.ups.append(UpScale(2 * channels, pixel_shuffle_factor=2))
            channels = channels // 2
            self.decoders.append(BlockStack(NAFBlock, num))
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

    def get_config(self) -> dict:
        """Add upscale factor to the config"""
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "middle_block_num": self.middle_block_num,
                "encoder_block_nums": self.encoder_block_nums,
                "decoder_block_nums": self.decoder_block_nums,
            }
        )
        return config
