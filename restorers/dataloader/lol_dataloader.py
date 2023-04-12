import os
from glob import glob
from functools import partial
from typing import Dict, List, Union, Tuple

import tensorflow as tf

from .base import LowLightDatasetFactory, UnsupervisedLowLightDatasetFactory
from .base.commons import (
    read_image,
    random_unpaired_horiontal_flip,
    random_unpaired_vertical_flip,
)

_AUTOTUNE = tf.data.AUTOTUNE


def _define_lol_dataset_structure(
    dataset_path: str, val_split: float
) -> Dict[str, List[List[str]]]:
    low_light_images = sorted(glob(os.path.join(dataset_path, "our485/low/*")))
    enhanced_images = sorted(glob(os.path.join(dataset_path, "our485/high/*")))
    test_input_images = sorted(glob(os.path.join(dataset_path, "eval15/low/*")))
    test_enhanced_images = sorted(glob(os.path.join(dataset_path, "eval15/high/*")))
    num_data_points = len(low_light_images)
    num_train_images = int(num_data_points * (1 - val_split))
    train_input_images = low_light_images[:num_train_images]
    train_enhanced_images = enhanced_images[:num_train_images]
    val_input_images = low_light_images[num_train_images:]
    val_enhanced_images = enhanced_images[num_train_images:]
    return {
        "train": [train_input_images, train_enhanced_images],
        "val": [val_input_images, val_enhanced_images],
        "test": [test_input_images, test_enhanced_images],
    }


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
        data_paths = _define_lol_dataset_structure(dataset_path, val_split)
        self.train_input_images, self.train_enhanced_images = data_paths["train"]
        self.val_input_images, self.val_enhanced_images = data_paths["val"]
        self.test_input_images, self.test_enhanced_images = data_paths["test"]


class UnsupervisedLOLDataLoader(UnsupervisedLowLightDatasetFactory):
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

    def define_dataset_structure(self, dataset_path: str, val_split: float) -> None:
        data_paths = _define_lol_dataset_structure(dataset_path, val_split)
        self.train_input_images, self.train_enhanced_images = data_paths["train"]
        self.val_input_images, self.val_enhanced_images = data_paths["val"]
        self.test_input_images, self.test_enhanced_images = data_paths["test"]
        if self.train_on_all_images:
            self.train_input_images = (
                self.train_input_images + self.train_enhanced_images
            )
            self.val_input_images = self.val_input_images + self.val_enhanced_images
