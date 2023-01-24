from typing import Dict

import tensorflow as tf


class DeepCurveEstimationLayer(tf.keras.layers.Layer):
    def __init__(
        self, num_intermediate_filters: int, num_iterations: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_intermediate_filters = num_intermediate_filters
        self.num_iterations = num_iterations

        self.convolution_1 = tf.keras.layers.Conv2D(
            filters=self.num_intermediate_filters,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
        self.convolution_2 = tf.keras.layers.Conv2D(
            filters=self.num_intermediate_filters,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
        self.convolution_3 = tf.keras.layers.Conv2D(
            filters=self.num_intermediate_filters,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
        self.convolution_4 = tf.keras.layers.Conv2D(
            filters=self.num_intermediate_filters,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
        self.convolution_5 = tf.keras.layers.Conv2D(
            filters=self.num_intermediate_filters * 2,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
        self.convolution_6 = tf.keras.layers.Conv2D(
            filters=self.num_intermediate_filters * 2,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
        self.convolution_7 = tf.keras.layers.Conv2D(
            filters=self.num_iterations * 3,
            kernel_size=(3, 3),
            padding="same",
            activation="tanh",
        )

    def call(self, inputs, training=None, mask=None):
        out_1 = self.convolution_1(inputs)
        out_2 = self.convolution_2(out_1)
        out_3 = self.convolution_3(out_2)
        out_4 = self.convolution_4(out_3)
        out_5 = self.convolution_5(tf.concat([out_3, out_4], axis=-1))
        out_6 = self.convolution_6(tf.concat([out_2, out_5], axis=-1))
        alphas_stacked = self.convolution_7(tf.concat([out_1, out_6], axis=-1))
        return alphas_stacked

    def get_config(self) -> Dict:
        return {
            "num_intermediate_filters": self.num_intermediate_filters,
            "num_iterations": self.num_iterations,
        }
