import unittest

import tensorflow as tf

from restorers.losses import SpatialConsistencyLoss


class SpatialConsistencyLossTest(unittest.TestCase):
    def test_spatial_constancy_loss(self) -> None:
        x = tf.ones((1, 256, 256, 3))
        self.assertEqual(SpatialConsistencyLoss()(x, x).numpy().item(), 0.0)
