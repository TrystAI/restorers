import shutil
import unittest

from restorers.dataloader import (
    LOLDataLoader,
    UnsupervisedLOLDataLoader,
    MITAdobe5KDataLoader,
)


class LowLightDataLoaderTester(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.image_size = 128
        self.bit_depth = 8
        self.val_split = 0.2

    def test_lol_dataloader(self) -> None:
        data_loader = LOLDataLoader(
            image_size=self.image_size,
            bit_depth=self.bit_depth,
            val_split=self.val_split,
            visualize_on_wandb=False,
            dataset_artifact_address="ml-colabs/mirnet-v2/lol-dataset:v0",
        )
        self.assertEqual(len(data_loader), 485)
        train_dataset, val_dataset, viz_dataset = data_loader.get_datasets(batch_size=1)
        x, y = next(iter(train_dataset))
        self.assertEqual(
            tuple(train_dataset.element_spec[0].shape),
            (1, self.image_size, self.image_size, 3),
        )
        self.assertEqual(
            tuple(train_dataset.element_spec[1].shape),
            (1, self.image_size, self.image_size, 3),
        )
        self.assertEqual(x.shape, (1, self.image_size, self.image_size, 3))
        self.assertEqual(y.shape, (1, self.image_size, self.image_size, 3))
        x, y = next(iter(val_dataset))
        self.assertEqual(
            tuple(val_dataset.element_spec[0].shape),
            (1, self.image_size, self.image_size, 3),
        )
        self.assertEqual(
            tuple(val_dataset.element_spec[1].shape),
            (1, self.image_size, self.image_size, 3),
        )
        self.assertEqual(x.shape, (1, self.image_size, self.image_size, 3))
        self.assertEqual(y.shape, (1, self.image_size, self.image_size, 3))
        x, y = next(iter(viz_dataset))
        self.assertEqual(
            tuple(val_dataset.element_spec[0].shape),
            (1, self.image_size, self.image_size, 3),
        )
        self.assertEqual(
            tuple(val_dataset.element_spec[1].shape),
            (1, self.image_size, self.image_size, 3),
        )
        self.assertEqual(x.shape, (1, self.image_size, self.image_size, 3))
        self.assertEqual(y.shape, (1, self.image_size, self.image_size, 3))
        shutil.rmtree("./artifacts")

    def test_unsupervised_lol_dataloader(self) -> None:
        data_loader = UnsupervisedLOLDataLoader(
            image_size=self.image_size,
            bit_depth=self.bit_depth,
            val_split=self.val_split,
            visualize_on_wandb=False,
            dataset_artifact_address="ml-colabs/mirnet-v2/lol-dataset:v0",
        )
        self.assertEqual(len(data_loader), 485)
        train_dataset, val_dataset, viz_dataset = data_loader.get_datasets(batch_size=1)
        x = next(iter(train_dataset))
        self.assertEqual(
            tuple(train_dataset.element_spec.shape),
            (1, self.image_size, self.image_size, 3),
        )
        self.assertEqual(x.shape, (1, self.image_size, self.image_size, 3))
        x = next(iter(val_dataset))
        self.assertEqual(
            tuple(val_dataset.element_spec.shape),
            (1, self.image_size, self.image_size, 3),
        )
        self.assertEqual(x.shape, (1, self.image_size, self.image_size, 3))
        x = next(iter(viz_dataset))
        self.assertEqual(x.shape, (1, self.image_size, self.image_size, 3))
        shutil.rmtree("./artifacts")

    def test_mit_adobe_5k_dataloader(self) -> None:
        data_loader = MITAdobe5KDataLoader(
            image_size=self.image_size,
            bit_depth=self.bit_depth,
            val_split=self.val_split,
            visualize_on_wandb=False,
            dataset_artifact_address="ml-colabs/mirnet-v2/mit-adobe-5k:v1",
        )
        self.assertEqual(len(data_loader), 5000)
        train_dataset, val_dataset, viz_dataset = data_loader.get_datasets(batch_size=1)
        x, y = next(iter(train_dataset))
        self.assertEqual(
            tuple(train_dataset.element_spec[0].shape),
            (1, self.image_size, self.image_size, 3),
        )
        self.assertEqual(
            tuple(train_dataset.element_spec[1].shape),
            (1, self.image_size, self.image_size, 3),
        )
        self.assertEqual(x.shape, (1, self.image_size, self.image_size, 3))
        self.assertEqual(y.shape, (1, self.image_size, self.image_size, 3))
        x, y = next(iter(val_dataset))
        self.assertEqual(
            tuple(val_dataset.element_spec[0].shape),
            (1, self.image_size, self.image_size, 3),
        )
        self.assertEqual(
            tuple(val_dataset.element_spec[1].shape),
            (1, self.image_size, self.image_size, 3),
        )
        self.assertEqual(x.shape, (1, self.image_size, self.image_size, 3))
        self.assertEqual(y.shape, (1, self.image_size, self.image_size, 3))
        x, y = next(iter(viz_dataset))
        self.assertEqual(
            tuple(val_dataset.element_spec[0].shape),
            (1, self.image_size, self.image_size, 3),
        )
        self.assertEqual(
            tuple(val_dataset.element_spec[1].shape),
            (1, self.image_size, self.image_size, 3),
        )
        self.assertEqual(x.shape, (1, self.image_size, self.image_size, 3))
        self.assertEqual(y.shape, (1, self.image_size, self.image_size, 3))
        shutil.rmtree("./artifacts")
