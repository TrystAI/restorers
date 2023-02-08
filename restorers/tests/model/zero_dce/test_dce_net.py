import unittest

import tensorflow as tf

from restorers.model.zero_dce import (
    DeepCurveEstimationLayer,
    FastDeepCurveEstimationLayer,
    ZeroDCE,
    FastZeroDce,
)
from restorers.model.zero_dce.dw_conv import DepthwiseSeparableConvolution


class ZeroDCETest(unittest.TestCase):
    def test_dce_layer(self) -> None:
        x = tf.ones((1, 256, 256, 3))
        dce_layer = DeepCurveEstimationLayer(
            num_intermediate_filters=32, num_iterations=8
        )
        y = dce_layer(x)
        self.assertEqual(y.shape, (1, 256, 256, 3 * dce_layer.num_iterations))

    def test_zero_dce(self) -> None:
        x = tf.ones((1, 256, 256, 3))
        model = ZeroDCE(num_intermediate_filters=32, num_iterations=8)
        output = model(x)
        self.assertEqual(output.shape, (1, 256, 256, 3))


class FastZeroDCETest(unittest.TestCase):
    def test_dw_conv(self) -> None:
        x = tf.ones((1, 256, 256, 3))
        dw_conv_layer = DepthwiseSeparableConvolution(
            intermediate_channels=3, output_channels=32
        )
        y = dw_conv_layer(x)
        self.assertEqual(y.shape, (1, 256, 256, dw_conv_layer.output_channels))

    def test_dce_layer(self) -> None:
        x = tf.ones((1, 256, 256, 3))
        dce_layer = FastDeepCurveEstimationLayer(
            num_intermediate_filters=32, num_iterations=8
        )
        y = dce_layer(x)
        self.assertEqual(y.shape, (1, 256, 256, 3))

    def test_zero_dce(self) -> None:
        x = tf.ones((1, 256, 256, 3))
        model = FastZeroDce(num_intermediate_filters=32, num_iterations=8)
        output = model(x)
        self.assertEqual(output.shape, (1, 256, 256, 3))
