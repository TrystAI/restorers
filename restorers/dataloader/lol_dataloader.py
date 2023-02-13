import os
from glob import glob
from typing import Union
from functools import partial

import tensorflow as tf

from .base import LowLightDatasetFactory
from .base.commons import read_image, unsupervised_random_horizontal_flip, unsupervised_random_vertical_flip

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


class UnsupervisedLOLDataLoader(LOLDataLoader):
    def __init__(
        self,
        image_size,
        bit_depth,
        val_split,
        visualize_on_wandb,
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

    def random_crop(self, input_image):
        # Check whether the image size is smaller than the original image
        image_size = tf.minimum(self.image_size, tf.shape(input_image)[0])

        # Apply random crop to the input image and split the stack
        cropped_image = tf.image.random_crop(input_image, (image_size, image_size, 3))

        # Ensuring the dataset tensor_spec is not None is the spatial dimensions
        cropped_image.set_shape([self.image_size, self.image_size, 3])

        return cropped_image

    def resize(self, input_image):
        # Check whether the image size is smaller than the original image
        image_size = tf.minimum(self.image_size, tf.shape(input_image)[0])

        # resize the image
        resized_image = tf.image.resize(input_image, size=[image_size, image_size])

        # Ensuring the dataset tensor_spec is not None is the spatial dimensions
        resized_image.set_shape([self.image_size, self.image_size, 3])

        return resized_image

    def load_image(self, input_image_path, apply_crop):
        # Read the image off the file path.
        input_image = read_image(input_image_path, self.normalization_factor)

        # Apply random cropping based on the boolean flag.
        input_image = (
            self.random_crop(input_image) if apply_crop else self.resize(input_image)
        )

        return input_image

    def build_dataset(self, input_images, batch_size, apply_crop, apply_augmentations):
        # Build a `tf.data.Dataset` from the filenames.
        dataset = tf.data.Dataset.from_tensor_slices((input_images))

        # Build the mapping function and apply it to the dataset.
        map_fn = partial(self.load_image, apply_crop=apply_crop)
        dataset = dataset.map(map_fn, num_parallel_calls=_AUTOTUNE)

        # Apply augmentations.
        if apply_augmentations:
            dataset = dataset.map(unsupervised_random_horizontal_flip, num_parallel_calls=_AUTOTUNE)
            dataset = dataset.map(unsupervised_random_vertical_flip, num_parallel_calls=_AUTOTUNE)

        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset.prefetch(_AUTOTUNE)

    def get_datasets(self, batch_size):
        train_dataset = self.build_dataset(
            input_images=self.train_input_images,
            batch_size=batch_size,
            apply_crop=True,
            apply_augmentations=True,
        )
        val_dataset = self.build_dataset(
            input_images=self.val_input_images,
            batch_size=batch_size,
            apply_crop=False,
            apply_augmentations=False,
        )
        return train_dataset, val_dataset
