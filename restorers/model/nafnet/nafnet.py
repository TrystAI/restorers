import tensorflow as tf
from tensorflow import keras
from .nafblock import NAFBlock


class PixelShuffle(keras.layers.Layer):
    def __init__(self, upscale_factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upscale_factor = upscale_factor

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.upscale_factor)


class NAFNet(keras.models.Model):
    def __init__(
        self,
        filters=16,
        middle_block_num=1,
        encoder_block_nums=(1, 1, 1, 1),
        decoder_block_nums=(1, 1, 1, 1),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
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
                f" and {len(decoder_block_nums)} were passed"
            )

        channels = filters
        for num in encoder_block_nums:
            self.encoders.append(
                keras.models.Sequential([NAFBlock(channels) for _ in range(num)])
            )
            self.downs.append(
                keras.layers.Conv2D(2 * channels, kernel_size=2, strides=2)
            )
            channels *= 2

        self.middle_blocks = keras.models.Sequential(
            [NAFBlock(channels) for _ in range(middle_block_num)]
        )

        for num in decoder_block_nums:
            self.ups.append(
                keras.models.Sequential(
                    [
                        keras.layers.Conv2D(
                            2 * channels, kernel_size=2, strides=1, bias=False
                        ),
                        PixelShuffle(2),
                    ]
                )
            )
            channels = channels // 2
            self.decoders.append(
                keras.models.Sequential([[NAFBlock(channels) for _ in range(num)]])
            )

    def build(self, input_shape):
        input_channels = input_shape[-1]
        self.ending = keras.layers.Conv2DTranspose(
            filters=input_channels, kernel_size=3, padding="same"
        )

    def call(self, inputs):

        x = self.intro(inputs)

        encoder_outputs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encoder_outputs.append(x)
            x = down(x)

        x = self.middle_blocks(x)

        for decoder, up, encoder_output in zip(
            self.decoders, self.ups, encoder_outputs
        ):
            x = up(x)
            # Residual connection of encoder blocks with decoder blocks
            x = x + encoder_output
            x = decoder(x)

        x = self.ending(x)
        # Residual connection of inputs with output
        x = x + inputs

        return x
