import tensorflow as tf


class PixelShuffle(tf.keras.layers.Layer):
    """
    PixelShuffle Layer

    Given input of size (H,W,C), it will generate an output
    of size

    $$(H \cdot f, W \cdot f, \\frac{c}{f^2})$$

    Where $c$ is channels and $f$ is pixel_shuffle_factor

    While giving input, make sure that $f^2$ divides $c$.

    Wrapper Class for tf.nn.depth_to_space
    Reference: https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space

    Args:
        upscale_factor (int): the factor by which the input's spatial dimensions will be scaled.
    """

    def __init__(self, upscale_factor: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.upscale_factor = upscale_factor

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return tf.nn.depth_to_space(inputs, self.upscale_factor)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"upscale_factor": self.upscale_factor})
        return config


class UpScale(tf.keras.layers.Layer):
    """
    UpScale Layer

    Given channels and pixel_shuffle_factor as input, it will generate an output
    of size

    $$(H \cdot f, W \cdot f, \\frac{c}{f^2})$$

    Where $c$ is channels and $f$ is pixel_shuffle_factor

    While giving input, make sure that $f^2$ divides $c$.

    Args:
        channels (int): number of channels in the input.
        pixel_shuffle_factor (int): the factor by which the input's spatial dimensions will be scaled.
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

        self.conv = tf.keras.layers.Conv2D(
            channels, kernel_size=1, strides=1, use_bias=False
        )
        self.pixel_shuffle = PixelShuffle(pixel_shuffle_factor)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return self.pixel_shuffle(self.conv(inputs))

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "pixel_shuffle_factor": self.pixel_shuffle_factor,
            }
        )
        return config