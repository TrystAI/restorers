from typing import Optional

import tensorflow as tf


class DownSample(tf.keras.layers.Layer):
    def __init__(
        self,
        channels: int,
        channel_factor: float,
        name: str = "DownSample Block",
        *args,
        **kwargs
    ) -> None:
        super().__init__(name=name, *args, **kwargs)

        self.channels = channels
        self.channel_factor = channel_factor

        self.downsample = tf.keras.Sequential(
            [
                tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(
                    filters=int(self.channels * self.channel_factor),
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    use_bias=False,
                ),
            ]
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        return self.downsample(inputs)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "channel_factor": self.channel_factor,
            }
        )
        return config


class UpSample(tf.keras.layers.Layer):
    def __init__(
        self,
        channels: int,
        channel_factor: float,
        name: str = "UpSample Block",
        *args,
        **kwargs
    ) -> None:
        super().__init__(name=name, *args, **kwargs)

        self.channels = channels
        self.channel_factor = channel_factor

        self.downsample = tf.keras.Sequential(
            [
                tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear"),
                tf.keras.layers.Conv2D(
                    filters=int(self.channels // self.channel_factor),
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    use_bias=False,
                ),
            ]
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        return self.downsample(inputs)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "channel_factor": self.channel_factor,
            }
        )
        return config


class SkipUpSample(tf.keras.layers.Layer):
    def __init__(
        self,
        channels: int,
        channel_factor: float,
        name: str = "SkipUpSample Block",
        *args,
        **kwargs
    ) -> None:
        super().__init__(name=name, *args, **kwargs)

        self.channels = channels
        self.channel_factor = channel_factor

        self.skipupsample = tf.keras.Sequential(
            [
                tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear"),
                tf.keras.layers.Conv2D(
                    filters=int(self.channels // self.channel_factor),
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    use_bias=False,
                ),
            ]
        )

    def call(
        self, inputs: tf.Tensor, residual: tf.Tensor, training: Optional[bool] = None
    ) -> tf.Tensor:
        processed_input = self.skipupsample(inputs)
        return processed_input + residual

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "channel_factor": self.channel_factor,
            }
        )
        return config
