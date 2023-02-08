import unittest

import tensorflow as tf

from restorers.model.nafnet import NAFBlock, SimpleGate, SimplifiedChannelAttention


class NAFBlockTest(unittest.TestCase):
    def test_nafblock(self) -> None:
        input_shape = (1, 256, 256, 3)
        x = tf.ones(input_shape)
        nafblock = NAFBlock()
        y = nafblock(x)
        self.assertEqual(y.shape, x.shape)


class SimplifiedChannelAttentionTest(unittest.TestCase):
    def test_sca(self) -> None:
        input_shape = (1, 256, 256, 3)
        x = tf.ones(input_shape)
        sca = SimplifiedChannelAttention(input_shape[-1])
        y = sca(x)
        self.assertEqual(y.shape, x.shape)


class SimpleGateTest(unittest.TestCase):
    def setUp(self):
        self.factors = [2, 3]

    def test_simple_gate(self) -> None:
        input_shape = (1, 256, 256, 3)
        for factor in self.factors:
            scaled_shape = list(input_shape)
            scaled_shape[-1] = input_shape[-1] * factor
            x = tf.ones(scaled_shape)
            print(factor)
            simple_gate = SimpleGate(factor)
            y = simple_gate(x)
            self.assertEqual(y.shape, input_shape)
