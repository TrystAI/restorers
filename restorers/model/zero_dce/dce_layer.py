import tensorflow as tf


class DeepCurveEstimationLayer(tf.keras.layers.Layer):
    def __init__(self, filters: int, num_iterations: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.num_iterations = num_iterations

        self.build_convolutions()

    def build_convolutions(self):
        self.convolution_1 = tf.keras.layers.Conv2D(
            self.filters, (3, 3), strides=(1, 1), activation="relu", padding="same"
        )
        self.convolution_2 = tf.keras.layers.Conv2D(
            self.filters, (3, 3), strides=(1, 1), activation="relu", padding="same"
        )
        self.convolution_3 = tf.keras.layers.Conv2D(
            self.filters, (3, 3), strides=(1, 1), activation="relu", padding="same"
        )
        self.convolution_4 = tf.keras.layers.Conv2D(
            self.filters, (3, 3), strides=(1, 1), activation="relu", padding="same"
        )
        self.convolution_5 = tf.keras.layers.Conv2D(
            self.filters, (3, 3), strides=(1, 1), activation="relu", padding="same"
        )
        self.convolution_6 = tf.keras.layers.Conv2D(
            self.filters, (3, 3), strides=(1, 1), activation="relu", padding="same"
        )
        self.convolution_out = tf.keras.layers.Conv2D(
            3 * self.num_iterations,
            (3, 3),
            strides=(1, 1),
            activation="tanh",
            padding="same",
        )

    def call(self, inputs: tf.Tensor):
        conv1_out = self.convolution_1(inputs)
        conv2_out = self.convolution_2(conv1_out)
        conv3_out = self.convolution_3(conv2_out)
        conv4_out = self.convolution_4(conv3_out)
        conv5_out = self.convolution_5(tf.concat([conv4_out, conv3_out], axis=-1))
        conv6_out = self.convolution_6(tf.concat([conv5_out, conv2_out], axis=-1))
        alphas_stacked = self.convolution_out(
            tf.concat([conv6_out, conv1_out], axis=-1)
        )
        return alphas_stacked

    def get_config(self):
        return {"filters": self.filters, "num_iterations": self.num_iterations}
