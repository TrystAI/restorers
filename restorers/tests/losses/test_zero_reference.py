import unittest

import tensorflow as tf

from restorers.losses.zero_reference import (
    color_constancy_loss,
    illumination_smoothness_loss,
    exposure_loss,
)


class ZeroReferenceLossesTest(unittest.TestCase):
    def test_color_constancy(self) -> None:
        x = tf.zeros((1, 256, 256, 3))
        self.assertEqual(color_constancy_loss(x).numpy().item(), 0.0)

    def test_exposure_loss(self) -> None:
        x = tf.zeros((1, 256, 256, 3))
        self.assertAlmostEqual(exposure_loss(x).numpy().item(), 0.36)

    def test_illumination_smoothness_loss(self) -> None:
        x = tf.zeros((1, 256, 256, 3))
        self.assertEqual(illumination_smoothness_loss(x).numpy().item(), 0.0)
