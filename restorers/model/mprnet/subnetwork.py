from typing import Optional, Type

import tensorflow as tf

from .attention import ChannelAttentionBlock
from .resize import DownSample


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_features: int,
        kernel_size: int,
        scale_features: int,
        apply_csff: bool,
        reduction: Optional[int] = 4,
        use_bias: bool = False,
        activation: Optional[Type[tf.keras.layers.Layer]] = tf.keras.layers.PReLU,
        name: str = "encoder",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(name=name, *args, **kwargs)

        self.num_features = num_features
        self.kernel_size = kernel_size
        self.scale_features = scale_features
        self.apply_csff = apply_csff
        self.reduction = reduction
        self.use_bias = use_bias
        self.activation = activation

        self.encoder_layer_1 = tf.keras.Sequential(
            [
                ChannelAttentionBlock(
                    num_features=self.num_features,
                    kernel_size=self.kernel_size,
                    reduction=self.reduction,
                    use_bias=self.use_bias,
                    activation=self.activation,
                    name=f"encoder_layer_1_{i}",
                )
                for i in range(2)
            ]
        )

        self.encoder_layer_2 = tf.keras.Sequential(
            [
                ChannelAttentionBlock(
                    num_features=self.num_features + self.scale_features,
                    kernel_size=self.kernel_size,
                    reduction=self.reduction,
                    use_bias=self.use_bias,
                    activation=self.activation,
                    name=f"encoder_layer_2_{i}",
                )
                for i in range(2)
            ]
        )

        self.encoder_layer_3 = tf.keras.Sequential(
            [
                ChannelAttentionBlock(
                    num_features=self.num_features + (2 * self.scale_features),
                    kernel_size=self.kernel_size,
                    reduction=self.reduction,
                    use_bias=self.use_bias,
                    activation=self.activation,
                    name=f"encoder_layer_3_{i}",
                )
                for i in range(2)
            ]
        )

        self.downsample_1 = DownSample(
            channels=self.num_features,
            channel_factor=self.scale_features,
            name="downsample_1",
        )
        self.downsample_2 = DownSample(
            channels=self.num_features + self.scale_features,
            channel_factor=self.scale_features,
            name="downsample_2",
        )

        if self.apply_csff:
            self.csff_encoder_1 = tf.keras.layers.Conv2D(
                filters=self.num_features,
                kernel_size=1,
                strides=1,
                padding="same",
                use_bias=self.use_bias,
                name="csff_encoder_1",
            )
            self.csff_encoder_2 = tf.keras.layers.Conv2D(
                filters=self.num_features + self.scale_features,
                kernel_size=1,
                strides=1,
                padding="same",
                use_bias=self.use_bias,
                name="csff_encoder_2",
            )
            self.csff_encoder_3 = tf.keras.layers.Conv2D(
                filters=self.num_features + (2 * self.scale_features),
                kernel_size=1,
                strides=1,
                padding="same",
                use_bias=self.use_bias,
                name="csff_encoder_3",
            )

            self.csff_decoder_1 = tf.keras.layers.Conv2D(
                filters=self.num_features,
                kernel_size=1,
                strides=1,
                padding="same",
                use_bias=self.use_bias,
                name="csff_decoder_1",
            )
            self.csff_decoder_2 = tf.keras.layers.Conv2D(
                filters=self.num_features + self.scale_features,
                kernel_size=1,
                strides=1,
                padding="same",
                use_bias=self.use_bias,
                name="csff_decoder_2",
            )
            self.csff_decoder_3 = tf.keras.layers.Conv2D(
                filters=self.num_features + (2 * self.scale_features),
                kernel_size=1,
                strides=1,
                padding="same",
                use_bias=self.use_bias,
                name="csff_decoder_3",
            )

    def call(
        self,
        inputs: tf.Tensor,
        encoder_output: Optional[tf.Tensor] = None,
        decoder_output: Optional[tf.Tensor] = None,
        training: Optional[bool] = False,
    ) -> tf.Tensor:
        enc1 = self.encoder_layer_1(inputs, training=training)
        if (encoder_output is not None) and (decoder_output is not None):
            enc1 = (
                enc1
                + self.csff_encoder_1(encoder_output[0])
                + self.csff_decoder_1(decoder_output[0])
            )

        down1 = self.downsample_1(enc1)
        assert down1.shape[1] == inputs.shape[1] // 2

        enc2 = self.encoder_layer_2(down1, training=training)
        if (encoder_output is not None) and (decoder_output is not None):
            enc2 = (
                enc2
                + self.csff_encoder_2(encoder_output[1])
                + self.csff_decoder_2(decoder_output[1])
            )

        down2 = self.downsample_2(enc2)

        enc3 = self.encoder_layer_3(down2, training=training)
        if (encoder_output is not None) and (decoder_output is not None):
            enc3 = (
                enc3
                + self.csff_encoder_3(encoder_output[2])
                + self.csff_decoder_3(decoder_output[2])
            )

        return [enc1, enc2, enc3]

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "num_features": self.num_features,
                "kernel_size": self.kernel_size,
                "scale_features": self.scale_features,
                "apply_csff": self.apply_csff,
                "reduction": self.reduction,
                "use_bias": self.use_bias,
                "activation": self.activation,
            }
        )
        return config
