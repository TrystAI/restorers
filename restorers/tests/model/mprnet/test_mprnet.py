import unittest

import tensorflow as tf

from restorers.model.mprnet.attention import (
    ChannelAttentionBlock,
    ChannelAttentionLayer,
    SupervisedAttentionBlock,
)
from restorers.model.mprnet.resize import DownSample, SkipUpSample, UpSample
from restorers.model.mprnet.subnetwork import Encoder


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

    def test_channel_attention_layer(self) -> None:
        x = tf.ones((1, 128, 128, 180))
        y = ChannelAttentionLayer(channels=180, reduction=4)(x)
        self.assertEqual(y.shape, (1, 128, 128, 180))

    def test_channel_attention_block(self) -> None:
        x = tf.ones((1, 128, 128, 180))
        y = ChannelAttentionBlock(num_features=180, kernel_size=3, reduction=4)(x)
        self.assertEqual(y.shape, (1, 128, 128, 180))

    def test_supervised_attention_block(self) -> None:
        x = tf.ones((1, 128, 128, 180))
        x_img = tf.ones((1, 128, 128, 3))
        y, y_img = SupervisedAttentionBlock(num_features=180, kernel_size=3)(x, x_img)
        self.assertEqual(y.shape, (1, 128, 128, 180))

    def test_encoder(self) -> None:
        x = tf.ones((1, 128, 128, 180))
        y = Encoder(num_features=180, kernel_size=3, scale_features=48, csff=False)(x)
        self.assertEqual(y.shape, (1, 128, 128, 180))
