import unittest

import tensorflow as tf

from mirnetv2.dataloader import LOLDataLoader, MITAdobe5KDataLoader
from mirnetv2.model.downsample import DownBlock, DownSampleBlock
from mirnetv2.model.mirnet import MirNetv2, RecursiveResidualGroup
from mirnetv2.model.mrb import MultiScaleResidualBlock
from mirnetv2.model.rcb import ContextBlock, ResidualContextBlock
from mirnetv2.model.skff import SelectiveKernelFeatureFusion
from mirnetv2.model.upsample import UpBlock, UpSampleBlock


class ModelTester(unittest.TestCase):
    def test_skff(self):
        x = tf.ones((1, 256, 256, 120))
        y = SelectiveKernelFeatureFusion(channels=120)([x, x])
        self.assertEqual(y.shape, (1, 256, 256, 120))

    def test_context_block(self):
        x = tf.ones((1, 256, 256, 80))
        y = ContextBlock(channels=80)(x)
        self.assertEqual(y.shape, (1, 256, 256, 80))

    def test_residual_context_block(self):
        x = tf.ones((1, 256, 256, 80))
        y = ResidualContextBlock(channels=80, groups=1)(x)
        self.assertEqual(y.shape, (1, 256, 256, 80))
        y = ResidualContextBlock(channels=80, groups=2)(x)
        self.assertEqual(y.shape, (1, 256, 256, 80))

    def test_down_block(self):
        x = tf.ones((1, 256, 256, 80))
        y = DownBlock(channels=80, channel_factor=1.5)(x)
        self.assertEqual(y.shape, (1, 128, 128, 120))

    def test_downsample_block(self):
        x = tf.ones((1, 256, 256, 80))
        y = DownSampleBlock(channels=80, scale_factor=2, channel_factor=1.5)(x)
        self.assertEqual(y.shape, (1, 128, 128, 120))

    def test_up_block(self):
        x = tf.ones((1, 128, 128, 180))
        y = UpBlock(channels=180, channel_factor=1.5)(x)
        self.assertEqual(y.shape, (1, 256, 256, 120))

    def test_upsample_block(self):
        x = tf.ones((1, 128, 128, 180))
        y = UpSampleBlock(channels=180, scale_factor=2, channel_factor=1.5)(x)
        self.assertEqual(y.shape, (1, 256, 256, 120))

    def test_mrb(self):
        x = tf.ones((1, 256, 256, 80))
        y = MultiScaleResidualBlock(channels=80, channel_factor=1.5, groups=1)(x)
        self.assertEqual(y.shape, (1, 256, 256, 80))

    def test_rrg(self):
        x = tf.ones((1, 256, 256, 80))
        y = RecursiveResidualGroup(
            channels=80, num_mrb_blocks=2, channel_factor=1.5, groups=1
        )(x)
        self.assertEqual(y.shape, (1, 256, 256, 80))
        y = RecursiveResidualGroup(
            channels=80, num_mrb_blocks=2, channel_factor=1.5, groups=2
        )(x)
        self.assertEqual(y.shape, (1, 256, 256, 80))
        y = RecursiveResidualGroup(
            channels=80, num_mrb_blocks=2, channel_factor=1.5, groups=4
        )(x)
        self.assertEqual(y.shape, (1, 256, 256, 80))

    def test_mirnet_v2(self):
        x = tf.ones((1, 256, 256, 3))
        y = MirNetv2(
            channels=80,
            channel_factor=1.5,
            num_mrb_blocks=2,
            add_residual_connection=True,
        )(x)
        self.assertEqual(y.shape, (1, 256, 256, 3))


class LowLightDataLoaderTester(unittest.TestCase):
    def test_lol_dataloader(self):
        data_loader = LOLDataLoader(
            image_size=128,
            bit_depth=8,
            val_split=0.2,
            visualize_on_wandb=False,
            dataset_artifact_address="ml-colabs/mirnet-v2/lol-dataset:v0",
        )
        self.assertEqual(len(data_loader), 485)
        train_dataset, val_dataset = data_loader.get_datasets(batch_size=1)
        x, y = next(iter(train_dataset))
        self.assertEqual(x.shape, (1, 128, 128, 3))
        self.assertEqual(y.shape, (1, 128, 128, 3))
        x, y = next(iter(val_dataset))
        self.assertEqual(x.shape, (1, 128, 128, 3))
        self.assertEqual(y.shape, (1, 128, 128, 3))

    def test_mit_adobe_5k_dataloader(self):
        data_loader = MITAdobe5KDataLoader(
            image_size=128,
            bit_depth=8,
            val_split=0.2,
            visualize_on_wandb=False,
            dataset_artifact_address="ml-colabs/mirnet-v2/mit-adobe-5k:v1",
        )
        self.assertEqual(len(data_loader), 5000)
        train_dataset, val_dataset = data_loader.get_datasets(batch_size=1)
        x, y = next(iter(train_dataset))
        self.assertEqual(x.shape, (1, 128, 128, 3))
        self.assertEqual(y.shape, (1, 128, 128, 3))
        x, y = next(iter(val_dataset))
        self.assertEqual(x.shape, (1, 128, 128, 3))
        self.assertEqual(y.shape, (1, 128, 128, 3))
