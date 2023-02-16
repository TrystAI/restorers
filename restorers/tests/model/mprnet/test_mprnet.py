import unittest

import tensorflow as tf

from restorers.model.mprnet.resize import DownSample, SkipUpSample, UpSample


class ModelTester(unittest.TestCase):
    def test_downsample_layer(self) -> None:
        x = tf.ones((1, 256, 256, 80))
        y = DownSample(channels=80, channel_factor=1.5)(x)
        self.assertEqual(y.shape, (1, 128, 128, 120))

    def test_upsample_layer(self) -> None:
        x = tf.ones((1, 128, 128, 180))
        y = UpSample(channels=180, channel_factor=1.5)(x)
        self.assertEqual(y.shape, (1, 256, 256, 120))

    def test_skipupsample_layer(self) -> None:
        x = tf.ones((1, 128, 128, 180))
        y = tf.zeros((1, 256, 256, 120))
        z = SkipUpSample(channels=180, channel_factor=1.5)(x, y)
        self.assertEqual(z.shape, (1, 256, 256, 120))
