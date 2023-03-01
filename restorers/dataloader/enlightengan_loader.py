import os
from glob import glob
from typing import Union

from .base import UnsupervisedDatasetFactory
from ..utils import fetch_wandb_artifact


class EnlightenGANDataLoader(UnsupervisedDatasetFactory):
    def __init__(
        self,
        image_size: int,
        bit_depth: int,
        val_split: float,
        use_low_light_images_only: bool = True,
        dataset_artifact_address: str = None,
    ) -> None:
        super().__init__(image_size, bit_depth, val_split)
        self.use_low_light_images_only = use_low_light_images_only
        self.dataset_artifact_address = dataset_artifact_address
        self.fetch_dataset(self.val_split, self.dataset_artifact_address)

    def fetch_dataset(self, val_split: float, dataset_artifact_address: str):
        artifact_dir = fetch_wandb_artifact(
            artifact_address=dataset_artifact_address, artifact_type="dataset"
        )
        train_a_paths = sorted(glob(os.path.join(artifact_dir, "trainA", "*")))
        train_b_paths = sorted(glob(os.path.join(artifact_dir, "trainB", "*")))
        input_image_paths = (
            train_a_paths
            if self.use_low_light_images_only
            else train_a_paths + train_b_paths
        )
        self.num_data_points = len(input_image_paths)
        num_train_images = int(self.num_data_points * (1 - val_split))
        self.train_input_images = input_image_paths[:num_train_images]
        self.val_input_images = input_image_paths[num_train_images:]
