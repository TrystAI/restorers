import unittest
import tensorflow as tf

from mirnetv2.model.skff import SelectiveKernelFeatureFusion


class ModelTester(unittest.TestCase):
    
    def test_skff(self):
        x = tf.ones((1, 256, 256, 120))
        y = SelectiveKernelFeatureFusion(channels=120)([x, x])
        self.assertEqual(y.shape, (1, 256, 256, 120))
