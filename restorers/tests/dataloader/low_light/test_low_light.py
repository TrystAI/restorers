import shutil
import unittest

import tensorflow as tf

from restorers.dataloader import (
    LOLDataLoader,
    UnsupervisedLoLDataloader,
    MITAdobe5KDataLoader,
    EnlightenGANDataLoader,
)


class LowLightDataLoaderTester(unittest.TestCase):
    def test_lol_dataloader(self) -> None:
        data_loader = LOLDataLoader(
            image_size=128,
            bit_depth=8,
            val_split=0.2,
            visualize_on_wandb=False,
            dataset_artifact_address="ml-colabs/dataset/LoL:v0",
        )
        self.assertEqual(len(data_loader), 485)
        train_dataset, val_dataset = data_loader.get_datasets(batch_size=1)
        x, y = next(iter(train_dataset))
        self.assertEqual(tuple(train_dataset.element_spec[0].shape), (1, 128, 128, 3))
        self.assertEqual(tuple(train_dataset.element_spec[1].shape), (1, 128, 128, 3))
        self.assertEqual(x.shape, (1, 128, 128, 3))
        self.assertEqual(y.shape, (1, 128, 128, 3))
        x, y = next(iter(val_dataset))
        self.assertEqual(tuple(val_dataset.element_spec[0].shape), (1, 128, 128, 3))
        self.assertEqual(tuple(val_dataset.element_spec[1].shape), (1, 128, 128, 3))
        self.assertEqual(x.shape, (1, 128, 128, 3))
        self.assertEqual(y.shape, (1, 128, 128, 3))
        shutil.rmtree("./artifacts")

    def test_unsupervised_lol_dataloader(self) -> None:
        data_loader = UnsupervisedLoLDataloader(
            image_size=128,
            bit_depth=8,
            val_split=0.2,
            dataset_artifact_address="ml-colabs/dataset/LoL:v0",
        )
        self.assertEqual(len(data_loader), 485)
        train_dataset, val_dataset = data_loader.get_datasets(batch_size=1)
        x = next(iter(train_dataset))
        self.assertEqual(tuple(train_dataset.element_spec.shape), (1, 128, 128, 3))
        self.assertEqual(x.shape, (1, 128, 128, 3))
        x = next(iter(val_dataset))
        self.assertEqual(tuple(val_dataset.element_spec.shape), (1, 128, 128, 3))
        self.assertEqual(x.shape, (1, 128, 128, 3))
        shutil.rmtree("./artifacts")

    def test_mit_adobe_5k_dataloader(self) -> None:
        data_loader = MITAdobe5KDataLoader(
            image_size=128,
            bit_depth=16,
            val_split=0.2,
            visualize_on_wandb=False,
            dataset_artifact_address="ml-colabs/mirnet-v2/mit-adobe-5k:v1",
        )
        self.assertEqual(len(data_loader), 5000)
        train_dataset, val_dataset = data_loader.get_datasets(batch_size=1)
        x, y = next(iter(train_dataset))
        self.assertEqual(tuple(train_dataset.element_spec[0].shape), (1, 128, 128, 3))
        self.assertEqual(tuple(train_dataset.element_spec[1].shape), (1, 128, 128, 3))
        self.assertEqual(x.shape, (1, 128, 128, 3))
        self.assertEqual(y.shape, (1, 128, 128, 3))
        x, y = next(iter(val_dataset))
        self.assertEqual(tuple(val_dataset.element_spec[0].shape), (1, 128, 128, 3))
        self.assertEqual(tuple(val_dataset.element_spec[1].shape), (1, 128, 128, 3))
        self.assertEqual(x.shape, (1, 128, 128, 3))
        self.assertEqual(y.shape, (1, 128, 128, 3))
        shutil.rmtree("./artifacts")

    def test_enlightengan_dataloader(self) -> None:
        data_loader = EnlightenGANDataLoader(
            image_size=128,
            bit_depth=8,
            val_split=0.2,
            use_low_light_images_only=False,
            dataset_artifact_address="ml-colabs/dataset/EnlightenGAN-Dataset:v0",
        )
        self.assertEqual(len(data_loader), 1930)
        train_dataset, val_dataset = data_loader.get_datasets(batch_size=1)
        x, y = next(iter(train_dataset))
        self.assertEqual(tuple(train_dataset.element_spec[0].shape), (1, 128, 128, 3))
        self.assertEqual(tuple(train_dataset.element_spec[1].shape), (1, 128, 128, 3))
        self.assertEqual(x.shape, (1, 128, 128, 3))
        self.assertEqual(y.shape, (1, 128, 128, 3))
        x, y = next(iter(val_dataset))
        self.assertEqual(tuple(val_dataset.element_spec[0].shape), (1, 128, 128, 3))
        self.assertEqual(tuple(val_dataset.element_spec[1].shape), (1, 128, 128, 3))
        self.assertEqual(x.shape, (1, 128, 128, 3))
        self.assertEqual(y.shape, (1, 128, 128, 3))
        shutil.rmtree("./artifacts")
