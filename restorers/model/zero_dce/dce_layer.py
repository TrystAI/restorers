from typing import Dict

import tensorflow as tf


class DeepCurveEstimationLayer(tf.keras.layers.Layer):
    """The Deep Curve Estimation layer (also referred to as the DCE-Net) implemented as a
    `tf.keras.layers.Layer`.

    The input to the DCE layer is a low-light image while the outputs are a set of pixel-wise
    curve parameter maps for corresponding higherorder curves. The DCE layer contains seven
    convolutional layers with symmetrical skip-connection. In the first six convolutional layers,
    each convolutional layer consists of 32 convolutional kernels of size 3×3 and stride 1 followed
    by the ReLU activation function. The last convolutional layer consists of 24 convolutional kernels
    of size 3×3 and stride 1 followed by the Tanh activation function, which produces 24 curve parameter
    maps for eight iterations, where each iteration requires three curve parameter maps for the three
    channels (i.e., RGB channels).

    ![](https://i.imgur.com/HtIg34W.png)

    Reference:

    1. [Zero-DCE: Zero-reference Deep Curve Estimation for Low-light Image Enhancement](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)
    2. [Zero-Reference Learning for Low-Light Image Enhancement (Supplementary Material)](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Guo_Zero-Reference_Deep_Curve_CVPR_2020_supplemental.pdf)
    3. [Official PyTorch implementation of Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/model.py)
    4. [Unofficial PyTorch implementation of Zero-DCE](https://github.com/bsun0802/Zero-DCE/blob/master/code/model.py)
    5. [Tensorflow implementation of Zero-DCE](https://github.com/tuvovan/Zero_DCE_TF)
    6. [Keras tutorial for implementing Zero-DCE](https://keras.io/examples/vision/zero_dce/#dcenet)

    Args:
        num_intermediate_filters (int): number of filters in the intermediate convolutional layers.
        num_iterations (int): number of iterations of enhancement.
    """

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

    def call(
        self, inputs: tf.Tensor, training=None, mask=None, *args, **kwargs
    ) -> tf.Tensor:
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
