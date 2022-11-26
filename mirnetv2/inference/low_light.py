import os
from typing import Optional, Union, List

import wandb
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm.autonotebook import tqdm

from .base import BaseInferer


class LowLightInferer(BaseInferer):
    def __init__(
        self,
        model: Optional[tf.keras.Model] = None,
        model_artifact_address: Optional[str] = None,
        psnr_max_val: float = 1,
        ssim_max_val: float = 1,
    ) -> None:
        super().__init__(model, model_artifact_address, psnr_max_val, ssim_max_val)

    def preprocess_image(self, image: Image):
        """Preprocesses the image for inference.
        Returns:
            A numpy array of shape (1, height, width, 3) preprocessed for inference.
        """
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)

    def postprocess_image(self, model_output: Union[np.array, tf.Tensor]):
        """Postprocesses the model output for inference.
        Returns:
            A list of PIL.Image.Image objects postprocessed for visualization.
        """
        model_output = model_output * 255.0
        model_output = model_output.clip(0, 255)
        images = []
        for idx in range(model_output.shape[0]):
            image = model_output[idx].reshape(
                (np.shape(model_output)[1], np.shape(model_output)[2], 3)
            )
            images.append(Image.fromarray(np.uint8(image)))
        return images

    def infer_from_files(
        self,
        low_light_image_files: List[str],
        ground_truth_image_files: Optional[List[str]] = None,
        log_on_wandb: bool = False,
        save_dir: Optional[str] = None,
    ):
        """Perform inference on a list of low light image files.
        Args:
            low_light_image_files (List[str]):
                List of paths to low light images.
            ground_truth_image_files (Optional[List[str]], optional):
                List of paths to ground truth images. Defaults to None.
            log_on_wandb (bool, optional):
                Log inference results on Weights & Biases. Defaults to False.
            save_dir (Optional[str], optional):
                Path to directory to save the inference results. Defaults to None.
        Usage:
            ```python
            inferer.infer_from_files(low_light_image_files=glob("path-to-images/*.png"))
            ```
        """
        if ground_truth_image_files is not None:
            assert len(low_light_image_files) == len(ground_truth_image_files)
        if log_on_wandb:
            columns = ["Low-Light-Image-File", "Low-Light-Image", "Predicted-Image"]
            columns = (
                columns
                + [
                    "Ground-Truth-Image-File",
                    "Ground-Truth-Image",
                    "Peak-Signal-to-Noise-Ratio",
                    "Structural-Similarity",
                ]
                if ground_truth_image_files is not None
                else columns
            )
            table = wandb.Table(columns=columns)
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for idx in tqdm(range(len(low_light_image_files))):
            (
                low_light_image,
                model_output,
                predicted_image,
            ) = self.single_image_file_inference(low_light_image_files[idx])
            if save_dir is not None:
                predicted_image.save(
                    os.path.join(save_dir, low_light_image_files[idx].split("/")[-1])
                )
            if log_on_wandb:
                data = [
                    low_light_image_files[idx].split("/")[-1],
                    wandb.Image(low_light_image),
                    wandb.Image(predicted_image),
                ]
                if ground_truth_image_files is not None:
                    preprocessed_ground_truth_image = self.preprocess_image(
                        Image.open(ground_truth_image_files[idx])
                    )
                    psnr = tf.image.psnr(
                        preprocessed_ground_truth_image, model_output, self.psnr_max_val
                    )
                    ssim = tf.image.ssim(
                        preprocessed_ground_truth_image, model_output, self.ssim_max_val
                    )
                    data = data + [
                        ground_truth_image_files[idx].split("/")[-1],
                        wandb.Image(Image.open(ground_truth_image_files[idx])),
                        psnr.numpy()[0],
                        ssim.numpy()[0],
                    ]
                table.add_data(*data)
        if log_on_wandb:
            wandb.log(
                {
                    "Inference-With-Ground-Truth"
                    if ground_truth_image_files is not None
                    else "Inference": table
                }
            )

    def infer_from_batch(
        self,
        low_light_image_batch: np.array,
        ground_truth_image_batch: Optional[np.array] = None,
        log_on_wandb: bool = False,
        save_dir: Optional[str] = None,
    ):
        """Perform inference on a batch of low light images.
        Args:
            low_light_image_batch (np.array):
                A batch of low light images as numpy arrays. The batch should not be pre-processed.
            ground_truth_image_batch (Optional[np.array], optional):
                A batch of ground truth images. Defaults to None.
            log_on_wandb (bool, optional):
                Log inference results on Weights & Biases. Defaults to False.
            save_dir (Optional[str], optional):
                Path to directory to save the inference results. Defaults to None.
        Usage:
            ```python
            low_light_image_files = glob("path-to-images/*.png")
            low_light_image_batch = np.array(
                [np.array(Image.open(file_path)) for file_path in low_light_image_files]
            )
            inferer.infer_from_batch(low_light_image_batch=low_light_image_batch)
            ```
        """
        if ground_truth_image_batch is not None:
            assert low_light_image_batch.shape[0] == ground_truth_image_batch.shape[0]
        if log_on_wandb:
            columns = ["Low-Light-Image", "Predicted-Image"]
            columns = (
                columns
                + [
                    "Ground-Truth-Image",
                    "Peak-Signal-to-Noise-Ratio",
                    "Structural-Similarity",
                ]
                if ground_truth_image_batch is not None
                else columns
            )
            table = wandb.Table(columns=columns)
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        (
            low_light_images,
            model_output_batch,
            post_processed_images,
        ) = self.single_image_batch_inference(low_light_image_batch)
        for idx in tqdm(range(len(low_light_images))):
            predicted_image = post_processed_images[idx]
            if save_dir is not None:
                predicted_image.save(os.path.join(save_dir, f"{idx}.jpg"))
            if log_on_wandb:
                data = [
                    wandb.Image(low_light_images[idx]),
                    wandb.Image(predicted_image),
                ]
                if ground_truth_image_batch is not None:
                    preprocessed_ground_truth_image = self.preprocess_image(
                        Image.fromarray(np.uint8(ground_truth_image_batch[idx]))
                    )
                    psnr = tf.image.psnr(
                        preprocessed_ground_truth_image,
                        model_output_batch[idx],
                        self.psnr_max_val,
                    )
                    ssim = tf.image.ssim(
                        preprocessed_ground_truth_image,
                        model_output_batch[idx],
                        self.ssim_max_val,
                    )
                    data = data + [
                        wandb.Image(
                            Image.fromarray(np.uint8(ground_truth_image_batch[idx]))
                        ),
                        psnr.numpy()[0],
                        ssim.numpy()[0],
                    ]
                table.add_data(*data)
        if log_on_wandb:
            wandb.log(
                {
                    "Inference-With-Ground-Truth"
                    if ground_truth_image_batch is not None
                    else "Inference": table
                }
            )
