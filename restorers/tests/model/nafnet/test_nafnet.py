import unittest

import tensorflow as tf

from restorers.model.nafnet import (
    NAFBlock,
    SimpleGate,
    SimplifiedChannelAttention,
    NAFNet,
    PixelShuffle,
)


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
            simple_gate = SimpleGate(factor)
            y = simple_gate(x)
            self.assertEqual(y.shape, input_shape)


class NAFNetTest(unittest.TestCase):
    def test_nafnet(self) -> None:
        input_shape = (1, 256, 256, 3)
        x = tf.ones(input_shape)
        nafnet = NAFNet()
        y = nafnet(x)
        self.assertEqual(y.shape, x.shape)


class PixelShuffleTest(unittest.TestCase):
    def setUp(self):
        self.factors = [2, 3]

    def test_pixelshuffle(self) -> None:
        input_shape = (1, 256, 256, 3)
        for factor in self.factors:
            scaled_shape = list(input_shape)
            scaled_shape[-1] = input_shape[-1] * factor * factor
            x = tf.ones(scaled_shape)
            ps = PixelShuffle(factor)
            y = ps(x)
            upscaled_input_shape = list(input_shape)
            upscaled_input_shape[1] = input_shape[1] * factor
            upscaled_input_shape[2] = input_shape[2] * factor
            self.assertEqual(y.shape, upscaled_input_shape)