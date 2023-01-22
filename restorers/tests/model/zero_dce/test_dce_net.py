import unittest

import tensorflow as tf

from restorers.model.zero_dce import DeepCurveEstimationNetwork, ZeroDCE


class ZeroDCETest(unittest.TestCase):
    def test_dce_net(self) -> None:
        x = tf.ones((1, 256, 256, 3))
        y = DeepCurveEstimationNetwork(num_intermediate_filters=32, num_iterations=8)(x)
        self.assertEqual(y.shape, (1, 256, 256, 3 * 8))

    def test_zero_dce(self) -> None:
        x = tf.ones((1, 256, 256, 3))
        y = ZeroDCE(num_intermediate_filters=32, num_iterations=8)(x)
        self.assertEqual(y.shape, (1, 256, 256, 3))
