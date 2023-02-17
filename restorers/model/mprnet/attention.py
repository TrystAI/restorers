from typing import Optional, Type

import tensorflow as tf


class ChannelAttentionLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        channels: int,
        reduction: Optional[int] = 4,
        use_bias: Optional[bool] = False,
        name: str = "Channel Attention Layer",
        *args,
        **kwargs
    ) -> None:
        super().__init__(name=name, *args, **kwargs)

        self.avg_pool = tf.keras.layers.AveragePooling2D((1, 1))
        self.stem = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=int(channels // reduction),
                    kernel_size=1,
                    padding="same",
                    use_bias=use_bias,
                ),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(
                    filters=channels,
                    kernel_size=1,
                    padding="same",
                    use_bias=use_bias,
                ),
                tf.keras.layers.Activation("sigmoid"),
            ],
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = self.avg_pool(inputs)
        x = self.stem(x)
        return inputs * x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "reduction": self.reduction,
                "use_bias": self.use_bias,
            }
        )
        return config


class ChannelAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_features: int,
        kernel_size: int,
        reduction: Optional[int] = 16,
        use_bias: Optional[bool] = False,
        activation: Optional[Type[tf.keras.layers.Activation]] = tf.keras.layers.PReLU,
        name: str = "Channel Attention Block",
        *args,
        **kwargs
    ) -> None:
        super().__init__(name=name, *args, **kwargs)

        self.num_features = num_features
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.use_bias = use_bias
        self.activation = activation

        self.stem = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.num_features,
                    kernel_size=self.kernel_size,
                    padding="same",
                    use_bias=self.use_bias,
                ),
                self.activation(),
                tf.keras.layers.Conv2D(
                    filters=self.num_features,
                    kernel_size=self.kernel_size,
                    padding="same",
                    use_bias=self.use_bias,
                ),
            ]
        )

        self.channel_attention = ChannelAttentionLayer(
            channels=self.num_features,
            reduction=self.reduction,
            use_bias=self.use_bias,
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = self.stem(inputs)
        x = self.channel_attention(x)
        x += inputs
        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "num_features": self.num_features,
                "kernel_size": self.kernel_size,
                "reduction": self.reduction,
                "use_bias": self.use_bias,
                "activation": self.activation,
            }
        )
        return config


class SupervisedAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_features: int,
        kernel_size: int,
        use_bias: Optional[bool] = False,
        name: str = "Supervised Attention Block",
        *args,
        **kwargs
    ) -> None:
        super().__init__(name=name, *args, **kwargs)

        self.num_features = num_features
        self.kernel_size = kernel_size
        self.use_bias = use_bias

        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=self.num_features,
            kernel_size=self.kernel_size,
            padding="same",
            use_bias=self.use_bias,
        )

        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=self.kernel_size,
            padding="same",
            use_bias=self.use_bias,
        )

        self.conv_layer_3 = tf.keras.layers.Conv2D(
            filters=self.num_features,
            kernel_size=self.kernel_size,
            padding="same",
            use_bias=self.use_bias,
        )

    def call(
        self,
        inputs: tf.Tensor,
        image_inputs: tf.Tensor,
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        x1 = self.conv_layer_1(inputs)
        img = self.conv_layer_2(inputs) + image_inputs
        x2 = tf.keras.activations.sigmoid((self.conv_layer_3(img)))
        x1 = x1 * x2
        return x1 + inputs, img

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "num_features": self.num_features,
                "kernel_size": self.kernel_size,
                "use_bias": self.use_bias,
            }
        )
        return config
