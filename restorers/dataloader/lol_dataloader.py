import os
from glob import glob
from functools import partial
from typing import Union, List, Tuple

import tensorflow as tf

from .base import LowLightDatasetFactory
from .base.commons import (
    read_image,
    unsupervised_random_horizontal_flip,
    unsupervised_random_vertical_flip,
)
from ..utils import fetch_wandb_artifact

_AUTOTUNE = tf.data.AUTOTUNE


class LOLDataLoader(LowLightDatasetFactory):
    def __init__(
        self,
        image_size: int,
        bit_depth: int,
        val_split: float,
        visualize_on_wandb: bool,
        dataset_artifact_address: Union[str, None] = None,
        dataset_url: Union[str, None] = None,
    ):
        super().__init__(
            image_size,
            bit_depth,
            val_split,
            visualize_on_wandb,
            dataset_artifact_address,
            dataset_url,
        )

    def define_dataset_structure(self, dataset_path, val_split):
        low_light_images = sorted(glob(os.path.join(dataset_path, "our485/low/*")))
        enhanced_images = sorted(glob(os.path.join(dataset_path, "our485/high/*")))
        self.test_low_light_images = sorted(
            glob(os.path.join(dataset_path, "eval15/low/*"))
        )
        self.test_enhanced_images = sorted(
            glob(os.path.join(dataset_path, "eval15/high/*"))
        )
        self.num_data_points = len(low_light_images)
        num_train_images = int(self.num_data_points * (1 - val_split))
        self.train_input_images = low_light_images[:num_train_images]
        self.train_enhanced_images = enhanced_images[:num_train_images]
        self.val_input_images = low_light_images[num_train_images:]
        self.val_enhanced_images = enhanced_images[num_train_images:]


class UnsupervisedLoLDataloader:
    def __init__(
        self,
        image_size: int,
        bit_depth: int,
        val_split: float,
        dataset_artifact_address: Union[str, None] = None,
    ) -> None:
        self.image_size = image_size
        self.bit_depth = bit_depth
        self.val_split = val_split
        self.dataset_artifact_address = dataset_artifact_address
        self.fetch_dataset()

    def load_data(self, image_path: str) -> tf.Tensor:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(
            images=image,
            size=[self.image_size, self.image_size],
        )
        image = image / ((2**self.bit_depth) - 1)
        return image

    def fetch_dataset(self):
        artifact_dir = fetch_wandb_artifact(
            artifact_address=self.dataset_artifact_address, artifact_type="dataset"
        )
        low_light_images = sorted(glob(os.path.join(artifact_dir, "our485/low/*")))
        enhanced_images = sorted(glob(os.path.join(artifact_dir, "our485/high/*")))
        self.test_low_light_images = sorted(
            glob(os.path.join(artifact_dir, "eval15/low/*"))
        )
        self.test_enhanced_images = sorted(
            glob(os.path.join(artifact_dir, "eval15/high/*"))
        )
        self.num_data_points = len(low_light_images)
        num_train_images = int(self.num_data_points * (1 - self.val_split))
        self.train_input_images = low_light_images[:num_train_images]
        self.train_enhanced_images = enhanced_images[:num_train_images]
        self.val_input_images = low_light_images[num_train_images:]
        self.val_enhanced_images = enhanced_images[num_train_images:]

    def __len__(self):
        return self.num_data_points

    def build_dataset(
        self, input_image: List[str], batch_size: int, apply_augmentations: bool
    ) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((input_image))
        dataset = dataset.map(self.load_data, num_parallel_calls=_AUTOTUNE)
        if apply_augmentations:
            dataset = dataset.map(
                unsupervised_random_horizontal_flip,
                num_parallel_calls=_AUTOTUNE,
            )
            dataset = dataset.map(
                unsupervised_random_vertical_flip,
                num_parallel_calls=_AUTOTUNE,
            )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset.prefetch(_AUTOTUNE)

    def get_datasets(self, batch_size: int) -> Tuple[tf.data.Dataset]:
        train_dataset = self.build_dataset(
            input_image=self.train_input_images,
            batch_size=batch_size,
            apply_augmentations=True,
        )
        val_dataset = self.build_dataset(
            input_image=self.val_input_images,
            batch_size=batch_size,
            apply_augmentations=False,
        )
        return train_dataset, val_dataset
