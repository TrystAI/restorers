import unittest

import tensorflow as tf

from restorers.losses import CharbonnierLoss


class CharbonnierLossTest(unittest.TestCase):
    def __init__(self) -> None:
        x = tf.zeros((1, 256, 256, 3))
        self.assertAlmostEqual(
            CharbonnierLoss(epsilon=1e-3)(x, x).numpy().item(), 0.001
        )
