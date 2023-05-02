from functools import partial
from abc import abstractmethod
from typing import List, Tuple, Union

import tensorflow as tf
import wandb
from PIL import Image
from absl import logging
from tqdm.autonotebook import tqdm

from .base_dataloader import DatasetFactory
from .commons import (
    read_image,
    random_unpaired_horiontal_flip,
    random_unpaired_vertical_flip,
)

from restorers.utils import fetch_wandb_artifact

_AUTOTUNE = tf.data.AUTOTUNE


class LowLightDatasetFactory(DatasetFactory):
    """Abstract base class for building dataset factories or dataloaders for
    supervised low-light enhancement.

    Abstract functions to be overriden are:

    - `define_dataset_structure(self, dataset_path: str, val_split: float) -> None`
        - Function to define the structure of the dataset.

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
        if visualize_on_wandb:
            self.table = wandb.Table(
                columns=["Image-ID", "Split", "Low-Light-Image", "Ground-Truth-Image"]
            )
        self.dataset_url = dataset_url
        self.dataset_artifact_address = dataset_artifact_address
        super().__init__(image_size, bit_depth, val_split, visualize_on_wandb)

    @abstractmethod
    def define_dataset_structure(self, dataset_path: str, val_split: float) -> None:
        """Abstract function to define the structure of the dataset."""
        raise NotImplementedError(
            f"{self.__class__.__name__ }.define_dataset_structure"
        )

    def __len__(self):
        return self.num_data_points

    def _create_data_table(self, low_light_images, enhanced_images, split):
        for idx in tqdm(
            range(len(low_light_images)),
            desc=f"Generating visualizations for {split} images",
        ):
            self.table.add_data(
                int(low_light_images[idx].split("/")[-1][:-4]),
                split,
                wandb.Image(Image.open(low_light_images[idx])),
                wandb.Image(Image.open(enhanced_images[idx])),
            )

    def sanity_tests(self):
        """This function is used to visualize the dataset on Weights & Biases and enable
        interactive exploratory analysis.
        """
        try:
            self._create_data_table(
                self.train_input_images, self.train_enhanced_images, split="Train"
            )
        except:
            logging.warning("Train Set not found.")

        try:
            self._create_data_table(
                self.val_input_images, self.val_enhanced_images, split="Validation"
            )
        except:
            logging.warning("Validation Set not found.")

        try:
            self._create_data_table(
                self.test_low_light_images, self.test_enhanced_images, split="Test"
            )
        except:
            logging.warning("Test Set not found.")

        wandb.log({f"Lol-Dataset": self.table})

    def fetch_dataset(self, val_split, visualize_on_wandb: bool):
        """Function to fetch the dataset from a URL or a Weights & Biases dataset artifact.
        This function also executes the sanity tests.
        """
        if self.dataset_url is not None:
            dataset_path = tf.keras.utils.get_file(
                fname="lol_dataset.zip",
                origin=self.dataset_url,
                extract=True,
                archive_format="zip",
            ).split(".zip")[0]
        elif self.dataset_artifact_address is not None:
            dataset_path = fetch_wandb_artifact(
                self.dataset_artifact_address, artifact_type="dataset"
            )
        else:
            raise ValueError(
                "Both dataset_url and dataset_artifact_address cannot be None"
            )
        self.define_dataset_structure(dataset_path=dataset_path, val_split=val_split)
        if visualize_on_wandb and wandb.run is not None:
            self.sanity_tests()


class UnsupervisedLowLightDatasetFactory(LowLightDatasetFactory):
    """Abstract base class for building dataset factories or dataloaders for
    unsupervised low-light enhancement.

    Abstract functions to be overriden are:

    - `define_dataset_structure(self, dataset_path: str, val_split: float) -> None`
        - Function to define the structure of the dataset.

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
        viz_dataset = self.build_dataset(
            input_images=self.val_input_images,
            batch_size=1,
            apply_crop=False,
            apply_augmentations=False,
        )
        return train_dataset, val_dataset, viz_dataset
