"""
CLI Usage:
train.py:
  --experiment_configs: path to config file.
    (default: 'None')
  --wandb_entity_name: Name of Weights & Biases Entity
  --wandb_job_type: Type of Weights & Biases Job
  --wandb_project_name: Name of Weights & Biases Project
Example of overriding default configs using the CLI:
train.py:
  --experiment_configs configs/low_light.py
  --experiment_configs.data_loader_configs.batch_size 16
  --experiment_configs.model_configs.num_residual_recursive_groups 4
  --experiment_configs.training_configs.learning_rate 2e-4
"""

import os
from glob import glob
from time import time
from typing import Callable, List

import numpy as np
import tensorflow as tf
import wandb
from absl import app, flags, logging
from ml_collections.config_flags import config_flags
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from wandb.keras import WandbMetricsLogger

from restorers.model.zero_dce import ZeroDCE
from restorers.utils import plot_results, scale_tensor

FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="wandb_project_name", default=None, help="Name of Weights & Biases Project"
)
flags.DEFINE_string(
    name="wandb_run_name", default=None, help="Name of Weights & Biases Run"
)
flags.DEFINE_string(
    name="wandb_entity_name", default=None, help="Name of Weights & Biases Entity"
)
flags.DEFINE_string(
    name="wandb_job_type", default=None, help="Type of Weights & Biases Job"
)
flags.DEFINE_string(
    name="model_artifact_address",
    default=None,
    help="The Weights & Biases artifact address for the model",
)
config_flags.DEFINE_config_file("experiment_configs")


def main(_) -> None:
    using_wandb = False
    if FLAGS.wandb_project_name is not None:
        try:
            wandb.init(
                project=FLAGS.wandb_project_name,
                name=FLAGS.wandb_run_name,
                entity=FLAGS.wandb_entity_name,
                job_type=FLAGS.wandb_job_type,
                config=FLAGS.experiment_configs.to_dict(),
            )
            using_wandb = True
        except:
            logging.error("Unable to initialize_device wandb run.")

    artifact = wandb.use_artifact(FLAGS.model_artifact_address, type="model")
    model_configs = artifact.logged_by().config["model_configs"]
    model_path = artifact.download()
    model = tf.keras.models.load_model(model_path, compile=False)

    artifact = wandb.use_artifact(
        FLAGS.experiment_configs.data_loader_configs.dataset_artifact_address,
        type="dataset",
    )
    dataset_dir = artifact.download()

    train_val_low_light_images = sorted(
        glob(os.path.join(dataset_dir, "our485", "low", "*.png"))
    )
    train_val_ground_truth_images = sorted(
        glob(os.path.join(dataset_dir, "our485", "high", "*.png"))
    )
    test_low_light_images = sorted(
        glob(os.path.join(dataset_dir, "eval15", "low", "*.png"))
    )
    test_ground_truth_images = sorted(
        glob(os.path.join(dataset_dir, "eval15", "high", "*.png"))
    )

    print(
        "Number of low-light images in train-val split:",
        len(train_val_low_light_images),
    )
    print(
        "Number of ground-truth images in train-val split:",
        len(train_val_ground_truth_images),
    )
    print("Number of low-light images in Eval15 split:", len(test_low_light_images))
    print(
        "Number of ground-truth images in Eval15 split:", len(test_ground_truth_images)
    )

    def preprocess_image(image: Image.Image) -> np.ndarray:
        """Preprocesses the image for inference.

        Returns:
            A numpy array of shape (1, height, width, 3) preprocessed for inference.
        """
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image.astype("float32") / (
            (2**FLAGS.experiment_configs.data_loader_configs.bit_depth) - 1
        )
        return np.expand_dims(image, axis=0)

    def postprocess_image(model_output) -> Image.Image:
        """Postprocesses the model output for inference.

        Returns:
            A list of PIL.Image.Image objects postprocessed for visualization.
        """
        model_output = model_output * (
            (2**FLAGS.experiment_configs.data_loader_configs.bit_depth) - 1
        )
        model_output = model_output.clip(
            0, int((2**FLAGS.experiment_configs.data_loader_configs.bit_depth) - 1)
        )
        image = model_output[0].reshape(
            (np.shape(model_output)[1], np.shape(model_output)[2], 3)
        )
        return Image.fromarray(np.uint8(image))

    def infer_and_visualize(
        low_light_image_file: str, ground_truth_image_file: str, model: Callable
    ):
        low_light_image = Image.open(low_light_image_file)
        ground_truth_image = Image.open(ground_truth_image_file)
        preprocessed_image = preprocess_image(low_light_image)
        start = time()
        preprocessed_ground_truth = preprocess_image(ground_truth_image)
        inference_time = time() - start
        model_output = model.predict(preprocessed_image, verbose=0)
        psnr = tf.image.psnr(
            scale_tensor(preprocessed_image), scale_tensor(model_output), max_val=1.0
        )
        ssim = tf.image.ssim(
            scale_tensor(preprocessed_image), scale_tensor(model_output), max_val=1.0
        )
        post_processed_image = postprocess_image(model_output)
        return (
            low_light_image,
            ground_truth_image,
            post_processed_image,
            psnr,
            ssim,
            inference_time,
        )

    table = wandb.Table(
        columns=[
            "Input-Image",
            "Ground-Truth",
            "Image-Enhanced-By-AutoContrast",
            "Image-Enhanced-By-ZeroDCE",
            "Peak-Signal-Noise-Ratio",
            "Structual-Similarity",
            "Inference-Time",
            "Dataset",
        ]
    )

    train_val_psnr, train_val_ssim = 0.0, 0.0
    for idx in tqdm(range(len(train_val_low_light_images))):
        (
            low_light_image,
            ground_truth_image,
            mirnet_enhanced_image,
            psnr,
            ssim,
            inference_time,
        ) = infer_and_visualize(
            train_val_low_light_images[idx], train_val_ground_truth_images[idx], model
        )
        autocontrast_enhanced_image = ImageOps.autocontrast(low_light_image)
        table.add_data(
            wandb.Image(low_light_image),
            wandb.Image(ground_truth_image),
            wandb.Image(autocontrast_enhanced_image),
            wandb.Image(mirnet_enhanced_image),
            psnr.numpy().item(),
            ssim.numpy().item(),
            inference_time,
            "LoL/Train-Val",
        )
        train_val_psnr += psnr.numpy().item()
        train_val_ssim += ssim.numpy().item()

    test_psnr, test_ssim = 0.0, 0.0
    for idx in tqdm(range(len(test_low_light_images))):
        (
            low_light_image,
            ground_truth_image,
            mirnet_enhanced_image,
            psnr,
            ssim,
            inference_time,
        ) = infer_and_visualize(
            test_low_light_images[idx], test_ground_truth_images[idx], model
        )
        autocontrast_enhanced_image = ImageOps.autocontrast(low_light_image)
        table.add_data(
            wandb.Image(low_light_image),
            wandb.Image(ground_truth_image),
            wandb.Image(autocontrast_enhanced_image),
            wandb.Image(mirnet_enhanced_image),
            psnr.numpy().item(),
            ssim.numpy().item(),
            inference_time,
            "LoL/Eval15",
        )
        test_psnr += psnr.numpy().item()
        test_ssim += ssim.numpy().item()

    wandb.log(
        {
            "Evaluation": table,
            "Train-Val/Peak-Signal-Noise-Ratio": train_val_psnr
            / len(train_val_low_light_images),
            "Train-Val/Structual-Similarity": train_val_ssim
            / len(train_val_low_light_images),
            "Eval15/Peak-Signal-Noise-Ratio": test_psnr / len(test_low_light_images),
            "Eval15/Structual-Similarity": test_ssim / len(test_low_light_images),
        }
    )
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
