import unittest

import tensorflow as tf

from restorers.model.zero_dce import DeepCurveEstimationLayer, ZeroDCE
from restorers.model.zero_dce.dce_layer import DepthwiseSeparableConvolution


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
        intermediate_outputs, output = model(x)
        self.assertEqual(output.shape, (1, 256, 256, 3))
        self.assertEqual(len(intermediate_outputs), model.num_iterations - 1)
        for out in intermediate_outputs:
            self.assertEqual(out.shape, (1, 256, 256, 3))


class ZeroDCE2Test(unittest.TestCase):
    def test_dw_conv(self) -> None:
        x = tf.ones((1, 256, 256, 3))
        dw_conv_layer = DepthwiseSeparableConvolution(
            input_channels=3, output_channels=32
        )
        y = dw_conv_layer(x)
        self.assertEqual(y.shape, (1, 256, 256, dw_conv_layer.output_channels))
