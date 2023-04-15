import os
from glob import glob
from typing import Optional

from .base import UnsupervisedLowLightDatasetFactory


class LargeLowLightDataset(UnsupervisedLowLightDatasetFactory):
    def __init__(
        self,
        image_size: int,
        bit_depth: int,
        val_split: float,
        visualize_on_wandb: bool,
        max_images: Optional[int] = None,
    ):
        super().__init__(image_size, bit_depth, val_split, visualize_on_wandb)

    def define_dataset_structure(self, dataset_path, val_split):
        image_paths = glob(os.path.join(dataset_path, "*"))
        self.num_data_points = len(image_paths)
        num_train_images = int(self.num_data_points * (1 - val_split))
        self.train_input_images = image_paths[:num_train_images]
        self.val_input_images = image_paths[num_train_images:]
