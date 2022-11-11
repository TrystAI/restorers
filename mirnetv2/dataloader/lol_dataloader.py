import os
from glob import glob

import tensorflow as tf
import wandb
from PIL import Image
from tqdm.autonotebook import tqdm

from .base import DatasetFactory


class LOLDataLoader(DatasetFactory):
    def __init__(self, image_size: int, bit_depth: int, val_split: float, visualize_on_wandb: bool):
        if visualize_on_wandb:
            self.table = wandb.Table(
                columns=["Image-ID", "Split", "Low-Light-Image", "Ground-Truth-Image"]
            )
        # This zip file contains the images from [LOw Light paired dataset (LOL)](https://drive.google.com/open?id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB)
        self.dataset_url = "https://github.com/soumik12345/enhance-me/releases/download/v0.1/lol_dataset.zip"
        # This Weights and Biases artifact contains the [LOw Light paired dataset (LOL)](https://drive.google.com/open?id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB)
        self.dataset_artifact_address = (
            "geekyrakshit/compressed-mirnet/lol-dataset:latest"
        )
        super().__init__(image_size, bit_depth, val_split, visualize_on_wandb)

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

    def fetch_dataset(self, val_split, visualize_on_wandb: bool):
        dataset_path = (
            tf.keras.utils.get_file(
                fname="lol_dataset.zip",
                origin=self.dataset_url,
                extract=True,
                archive_format="zip",
            ).split(".zip")[0]
            if wandb.run is None
            else wandb.use_artifact(
                self.dataset_artifact_address, type="dataset"
            ).download()
        )
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
        self.train_low_light_images = low_light_images[:num_train_images]
        self.train_enhanced_images = enhanced_images[:num_train_images]
        self.val_low_light_images = low_light_images[num_train_images:]
        self.val_enhanced_images = enhanced_images[num_train_images:]
        if visualize_on_wandb and wandb.run is not None:
            self._create_data_table(
                self.train_low_light_images, self.train_enhanced_images, split="Train"
            )
            self._create_data_table(
                self.val_low_light_images, self.val_enhanced_images, split="Validation"
            )
            self._create_data_table(
                self.test_low_light_images, self.test_enhanced_images, split="Test"
            )
            wandb.log({f"Lol-Dataset": self.table})
