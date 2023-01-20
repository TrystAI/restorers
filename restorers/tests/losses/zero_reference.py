import unittest

import tensorflow as tf

from restorers.losses.zero_reference import color_constancy


class ZeroReferenceLossesTest(unittest.TestCase):
    def test_color_constancy(self) -> None:
        x = tf.ones((1, 256, 256, 3))
        self.assertEqual(color_constancy(x).numpy().item(), 0.0)
