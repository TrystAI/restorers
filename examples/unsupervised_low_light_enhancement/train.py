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

import tensorflow as tf
import wandb
from absl import app, flags, logging
from low_light_config import get_config
from ml_collections.config_flags import config_flags
from wandb.keras import WandbMetricsLogger

from restorers.model.zero_dce import ZeroDCE
from restorers.utils import get_model_checkpoint_callback, initialize_device

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
flags.DEFINE_integer(
    name="num_batches_for_eval",
    default=2,
    help="Number of batches to be evaluated every epoch for visualization on Weights & Biases",
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

    tf.keras.utils.set_random_seed(FLAGS.experiment_configs.seed)

    strategy = initialize_device()
    batch_size = (
        FLAGS.experiment_configs.data_loader_configs.local_batch_size
        * strategy.num_replicas_in_sync
    )
    wandb.config.global_batch_size = batch_size

    def load_data(image_path: str) -> tf.Tensor:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(
            images=image,
            size=[
                FLAGS.experiment_configs.data_loader_configs.image_size,
                FLAGS.experiment_configs.data_loader_configs.image_size,
            ],
        )
        image = image / (
            (2**FLAGS.experiment_configs.data_loader_configs.bit_depth) - 1
        )
        return image

    def data_generator(low_light_images: tf.TypeSpec) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((low_light_images))
        dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset

    artifact = wandb.use_artifact(
        FLAGS.experiment_configs.data_loader_configs.dataset_artifact_address,
        type="dataset",
    )
    artifact_dir = artifact.download()

    train_low_light_images = sorted(
        glob(os.path.join(artifact_dir, "our485", "low", "*"))
    )
    num_train_images = int(
        (1 - FLAGS.experiment_configs.data_loader_configs.val_split)
        * len(train_low_light_images)
    )
    val_low_light_images = train_low_light_images[num_train_images:]
    train_low_light_images = train_low_light_images[:num_train_images]

    train_dataset = data_generator(train_low_light_images)
    val_dataset = data_generator(val_low_light_images)

    with strategy.scope():
        model = ZeroDCE(
            num_intermediate_filters=FLAGS.experiment_configs.model_configs.num_intermediate_filters,
            num_iterations=FLAGS.experiment_configs.model_configs.num_iterations,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=FLAGS.experiment_configs.training_configs.learning_rate,
            ),
            weight_exposure_loss=FLAGS.experiment_configs.training_configs.weight_exposure_loss,
            weight_color_constancy_loss=FLAGS.experiment_configs.training_configs.weight_color_constancy_loss,
            weight_illumination_smoothness_loss=FLAGS.experiment_configs.training_configs.weight_illumination_smoothness_loss,
        )

    callbacks = [
        get_model_checkpoint_callback(
            filepath="checkpoint",
            save_best_only=FLAGS.experiment_configs.training_configs.save_best_checkpoint_only,
            using_wandb=True,
        )
    ]
    callbacks.append(WandbMetricsLogger(log_freq="batch"))

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=FLAGS.experiment_configs.training_configs.epochs,
        callbacks=callbacks,
    )

    wandb.finish()


if __name__ == "__main__":
    app.run(main)
