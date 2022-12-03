import os
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Callable, Dict

import wandb
import numpy as np
from tqdm import tqdm
from PIL import Image

import tensorflow as tf


class BaseInferer(ABC):
    """Inference Class for an MirNet Model.

    Args:
        model (Optional[keras.Model], optional):
            The `tf.keras.Model` to be used for inference. Defaults to None.
            If None, the address of a Weights & Biases model artifact should be passed.
        model_artifact_address (Optional[str], optional):
            The address of the model artifact on Weights & Biases or a URL. Defaults to None.
            If None, a `tf.keras.Model` should be passed to `model`.
        psnr_max_val (float):
            The dynamic range of the pixel values in the images for calculation of
            peak signal-noise ratio. Default is 1.0.
        ssim_max_val (float):
            The dynamic range of the pixel values in the images for calculation of
            structural similarity. Default is 1.0.

    Usage:
        A `tf.keras.Model` can be passed to the `model` parameter.

        ```python
        model = tf.keras.models.load_model("path/to/model")

        inferer = Inferer(model=model)
        ```

        Optinally the address of a Weights & Biases model artifact
        can be passed to the parameter`model_artifact_address`.

        ```python
        inferer = Inferer(
            model_artifact_address="ml-colabs/compressed-mirnet/run_1poazfk5_model:latest"
        )
        ```

        Note:
            A Weights & Biases run is not necessary to be initialized
            in order to use a model artifact. However, if a run is initialized,
            the inference operation will be tracked in the lineage of the artifacts being used.
    """

    def __init__(
        self,
        model: Optional[tf.keras.Model] = None,
        model_artifact_address: Optional[str] = None,
        metrics: Dict[str, Callable] = {},
    ) -> None:
        self.model = model
        self.model_artifact_address = model_artifact_address
        self.metrics = metrics
        if self.model is None:
            self.initialize_model()

    def initialize_model(self) -> None:
        artifact = (
            wandb.Api().artifact(self.model_artifact_address, type="model")
            if wandb.run is None
            else wandb.use_artifact(self.model_artifact_address, type="model")
        )
        model_path = artifact.download()
        producer_run = artifact.logged_by()
        self.model_configs = producer_run.config["model_configs"]
        self.model = tf.keras.models.load_model(model_path, compile=False)

    @abstractmethod
    def preprocess_image(self, image: Image):
        raise NotImplementedError(f"{self.__class__.__name__ }.preprocess_image")

    @abstractmethod
    def postprocess_image(self, model_output: Union[np.array, tf.Tensor]):
        raise NotImplementedError(f"{self.__class__.__name__ }.postprocess_image")

    def single_image_file_inference(self, image_file):
        input_image = Image.open(image_file)
        preprocessed_image = self.preprocess_image(input_image)
        model_output = self.model.predict(preprocessed_image, verbose=0)
        predicted_image = self.postprocess_image(model_output)[0]
        return input_image, model_output, predicted_image

    def single_image_batch_inference(self, image_batch):
        input_images = [Image.fromarray(np.uint8(image)) for image in image_batch]
        preprocessed_low_light_images_batch = input_images.astype("float32") / 255.0
        model_output_batch = self.model.predict(
            preprocessed_low_light_images_batch, verbose=0
        )
        post_processed_images = self.postprocess_image(model_output_batch)
        return input_images, model_output_batch, post_processed_images

    def infer_from_files(
        self,
        input_image_files: List[str],
        ground_truth_image_files: Optional[List[str]] = None,
        log_on_wandb: bool = False,
        save_dir: Optional[str] = None,
    ):
        """Perform inference on a list of low light image files.
        Args:
            input_image_files (List[str]):
                List of paths to low light images.
            ground_truth_image_files (Optional[List[str]], optional):
                List of paths to ground truth images. Defaults to None.
            log_on_wandb (bool, optional):
                Log inference results on Weights & Biases. Defaults to False.
            save_dir (Optional[str], optional):
                Path to directory to save the inference results. Defaults to None.
        Usage:
            ```python
            inferer.infer_from_files(input_image_files=glob("path-to-images/*.png"))
            ```
        """
        if ground_truth_image_files is not None:
            assert len(input_image_files) == len(ground_truth_image_files)
        if log_on_wandb:
            columns = ["Low-Light-Image-File", "Low-Light-Image", "Predicted-Image"]
            columns = (
                columns
                + ["Ground-Truth-Image-File", "Ground-Truth-Image"]
                + list(self.metrics.keys())
                if ground_truth_image_files is not None
                else columns
            )
            table = wandb.Table(columns=columns)
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for idx in tqdm(range(len(input_image_files))):
            (
                input_image,
                model_output,
                predicted_image,
            ) = self.single_image_file_inference(input_image_files[idx])
            if save_dir is not None:
                predicted_image.save(
                    os.path.join(save_dir, input_image_files[idx].split("/")[-1])
                )
            if log_on_wandb:
                data = [
                    input_image_files[idx].split("/")[-1],
                    wandb.Image(input_image),
                    wandb.Image(predicted_image),
                ]
                if ground_truth_image_files is not None:
                    preprocessed_ground_truth_image = self.preprocess_image(
                        Image.open(ground_truth_image_files[idx])
                    )
                    metric_values = [
                        self.metrics[key](
                            preprocessed_ground_truth_image, model_output
                        ).numpy()[0]
                        for key in self.metrics.keys()
                    ]
                    data = data + [
                        ground_truth_image_files[idx].split("/")[-1],
                        wandb.Image(Image.open(ground_truth_image_files[idx])),
                        *metric_values,
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
        input_image_batch: np.array,
        ground_truth_image_batch: Optional[np.array] = None,
        log_on_wandb: bool = False,
        save_dir: Optional[str] = None,
    ):
        """Perform inference on a batch of low light images.
        Args:
            input_image_batch (np.array):
                A batch of low light images as numpy arrays. The batch should not be pre-processed.
            ground_truth_image_batch (Optional[np.array], optional):
                A batch of ground truth images. Defaults to None.
            log_on_wandb (bool, optional):
                Log inference results on Weights & Biases. Defaults to False.
            save_dir (Optional[str], optional):
                Path to directory to save the inference results. Defaults to None.
        Usage:
            ```python
            input_image_files = glob("path-to-images/*.png")
            input_image_batch = np.array(
                [np.array(Image.open(file_path)) for file_path in input_image_files]
            )
            inferer.infer_from_batch(input_image_batch=input_image_batch)
            ```
        """
        if ground_truth_image_batch is not None:
            assert input_image_batch.shape[0] == ground_truth_image_batch.shape[0]
        if log_on_wandb:
            columns = ["Low-Light-Image", "Predicted-Image"]
            columns = (
                columns
                + ["Ground-Truth-Image-File", "Ground-Truth-Image"]
                + list(self.metrics.keys())
                if ground_truth_image_batch is not None
                else columns
            )
            table = wandb.Table(columns=columns)
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        (
            input_images,
            model_output_batch,
            post_processed_images,
        ) = self.single_image_batch_inference(input_image_batch)
        for idx in tqdm(range(len(input_images))):
            predicted_image = post_processed_images[idx]
            if save_dir is not None:
                predicted_image.save(os.path.join(save_dir, f"{idx}.jpg"))
            if log_on_wandb:
                data = [
                    wandb.Image(input_images[idx]),
                    wandb.Image(predicted_image),
                ]
                if ground_truth_image_batch is not None:
                    preprocessed_ground_truth_image = self.preprocess_image(
                        Image.fromarray(np.uint8(ground_truth_image_batch[idx]))
                    )
                    metric_values = [
                        self.metrics[key](
                            preprocessed_ground_truth_image, model_output_batch[idx]
                        ).numpy()[0]
                        for key in self.metrics.keys()
                    ]
                    data = data + [
                        wandb.Image(
                            Image.fromarray(np.uint8(ground_truth_image_batch[idx]))
                        ),
                        *metric_values,
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
