import unittest

import tensorflow as tf

from restorers.model.nafnet import (
    NAFBlock
)


class NAFBlockTest(unittest.TestCase):

    def test_nafblock(self) -> None:
        input_shape = (1, 256, 256, 3)
        x = tf.ones(input_shape)
        nafblock = NAFBlock(input_shape[-1])
        y = nafblock(x)
        self.assertEqual(y.shape, x.shape)