import os
from glob import glob
from functools import partial
from typing import Union, Tuple, List

import tensorflow as tf

from .base import LowLightDatasetFactory
from .base.commons import (
    read_image,
    random_unpaired_horiontal_flip,
    random_unpaired_vertical_flip,
)

_AUTOTUNE = tf.data.AUTOTUNE


class LOLDataLoader(LowLightDatasetFactory):
    """DataLoader for the [LOL dataset](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset). This
    dataloader can be used to build datasets for training supervised low-light image enhancement models
    using the LOL Dataset.

    Usage:

    ```py
    # define dataloader for the LoL dataset
    data_loader = LOLDataLoader(
        # size of image crops on which we will train
        image_size=128,
        # bit depth of the images
        bit_depth=8,
        # fraction of images for validation
        val_split=0.2,
        # visualize the dataset on WandB or not
        visualize_on_wandb=True,
        # the wandb artifact address of the dataset,
        # this can be found from the `Usage` tab of
        # the aforemenioned weave panel
        dataset_artifact_address="ml-colabs/dataset/LoL:v0",
    )

    # call `get_datasets` on the `data_loader` to get
    # the TensorFlow datasets corresponding to the
    # training and validation splits
    datasets = data_loader.get_datasets(batch_size=2)
    ```

    ??? example "Examples"
        - [Training a supervised low-light enhancement model using MirNetv2](../../examples/train_mirnetv2).
        - [Training a supervised low-light enhancement model using NAFNet](../../examples/train_nafnet).

    Args:
        image_size (int): The image resolution.
        bit_depth (int): Bit depth of the images for normalization.
        val_split (float): The percentage of validation split.
        visualize_on_wandb (bool): Flag to visualize the dataset on wandb.
        dataset_artifact_address (Union[str, None]): The address of the dataset artifact on
            Weights & Biases.
        dataset_url (Union[str, None]): The URL of the dataset hosted on the web. This is not necessary
            in case `dataset_artifact_address` has been specified.
    """

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
    """Unsupervised dataLoader for the [LOL dataset](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset).
    This dataloader can be used to build datasets for training unsupervised low-light image enhancement models
    using the LOL Dataset.

    Usage:

    ```py
    # define unsupervised dataloader for the LoL dataset
    data_loader = UnsupervisedLOLDataLoader(
        # size of image crops on which we will train
        image_size=128,
        # bit depth of the images
        bit_depth=8,
        # fraction of images for validation
        val_split=0.2,
        # visualize the dataset on WandB or not
        visualize_on_wandb=True,
        # the wandb artifact address of the dataset,
        # this can be found from the `Usage` tab of
        # the aforemenioned weave panel
        dataset_artifact_address="ml-colabs/dataset/LoL:v0",
        # train on all images including low-light and ground-truth or not
        train_on_all_images=True,
    )

    # call `get_datasets` on the `data_loader` to get
    # the TensorFlow datasets corresponding to the
    # training and validation splits
    datasets = data_loader.get_datasets(batch_size=2)
    ```

    ??? example "Examples"
        - [Training an unsupervised low-light enhancement model using Zero-DCE](../../examples/train_zero_dce).
        - [Training an unsupervised low-light enhancement model using Fast Zero-DCE](../../examples/train_fast_zero_dce).

    Args:
        image_size (int): The image resolution.
        bit_depth (int): Bit depth of the images for normalization.
        val_split (float): The percentage of validation split.
        visualize_on_wandb (bool): Flag to visualize the dataset on wandb.
        dataset_artifact_address (Union[str, None]): The address of the dataset artifact on
            Weights & Biases.
        dataset_url (Union[str, None]): The URL of the dataset hosted on the web. This is not necessary
            in case `dataset_artifact_address` has been specified.
        train_on_all_images (bool): Use both input and ground-truth images for training or not.
    """

    def __init__(
        self,
        image_size: int,
        bit_depth: int,
        val_split: float,
        visualize_on_wandb: bool,
        dataset_artifact_address: Union[str, None] = None,
        dataset_url: Union[str, None] = None,
        train_on_all_images: bool = False,
    ):
        self.train_on_all_images = train_on_all_images
        super().__init__(
            image_size,
            bit_depth,
            val_split,
            visualize_on_wandb,
            dataset_artifact_address,
            dataset_url,
        )

    def define_dataset_structure(self, dataset_path, val_split):
        super().define_dataset_structure(dataset_path, val_split)
        if self.train_on_all_images:
            self.train_input_images = (
                self.train_input_images + self.train_enhanced_images
            )
            self.val_input_images = self.val_input_images + self.val_enhanced_images

    def random_crop(self, input_image: tf.Tensor) -> Tuple[tf.Tensor]:
        # Check whether the image size is smaller than the original image
        image_size = tf.minimum(self.image_size, tf.shape(input_image)[0])

        # Apply same *random* crop to the concantenated image and split the stack
        cropped_input_image = tf.image.random_crop(
            input_image, (image_size, image_size, 3)
        )

        # Ensuring the dataset tensor_spec is not None is the spatial dimensions
        cropped_input_image.set_shape([self.image_size, self.image_size, 3])

        return cropped_input_image

    def resize(self, input_image: tf.Tensor) -> Tuple[tf.Tensor]:
        # Check whether the image size is smaller than the original image
        image_size = tf.minimum(self.image_size, tf.shape(input_image)[0])

        resized_input_image = tf.image.resize(
            input_image,
            size=[image_size, image_size],
        )

        # Ensuring the dataset tensor_spec is not None is the spatial dimensions
        resized_input_image.set_shape([self.image_size, self.image_size, 3])

        return resized_input_image

    def load_image(self, input_image_path: str, apply_crop: bool):
        # Read the image off the file path.
        input_image = read_image(input_image_path, self.normalization_factor)

        # Apply random cropping based on the boolean flag.
        return self.random_crop(input_image) if apply_crop else self.resize(input_image)

    def build_dataset(
        self,
        input_images: List[str],
        batch_size: int,
        apply_crop: bool,
        apply_augmentations: bool,
    ) -> tf.data.Dataset:
        # Build a `tf.data.Dataset` from the filenames.
        dataset = tf.data.Dataset.from_tensor_slices(input_images)

        # Build the mapping function and apply it to the dataset.
        map_fn = partial(self.load_image, apply_crop=apply_crop)
        dataset = dataset.map(
            map_fn,
            num_parallel_calls=_AUTOTUNE,
        )

        # Apply augmentations.
        if apply_augmentations:
            dataset = dataset.map(
                random_unpaired_horiontal_flip,
                num_parallel_calls=_AUTOTUNE,
            )
            dataset = dataset.map(
                random_unpaired_vertical_flip,
                num_parallel_calls=_AUTOTUNE,
            )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset.prefetch(_AUTOTUNE)

    def get_datasets(self, batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
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
