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

from restorers.model.zero_dce import ZeroDCE, FastZeroDce
from restorers.dataloader import UnsupervisedLoLDataloader
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

    data_loader_configs = FLAGS.experiment_configs.data_loader_configs
    model_configs = FLAGS.experiment_configs.model_configs
    training_configs = FLAGS.experiment_configs.training_configs

    strategy = initialize_device()
    batch_size = data_loader_configs.local_batch_size * strategy.num_replicas_in_sync
    wandb.config.global_batch_size = batch_size

    data_loader = UnsupervisedLoLDataloader(
        image_size=data_loader_configs.image_size,
        bit_depth=data_loader_configs.bit_depth,
        val_split=data_loader_configs.val_split,
        dataset_artifact_address=data_loader_configs.dataset_artifact_address,
    )
    train_dataset, val_dataset = data_loader.get_datasets(batch_size=batch_size)

    with strategy.scope():
        model = (
            ZeroDCE(
                num_intermediate_filters=model_configs.num_intermediate_filters,
                num_iterations=model_configs.num_iterations,
                decoder_channel_factor=model_configs.decoder_channel_factor,
            )
            if not model_configs.use_faster_variant
            else FastZeroDce(
                num_intermediate_filters=model_configs.num_intermediate_filters,
                num_iterations=model_configs.num_iterations,
                decoder_channel_factor=model_configs.decoder_channel_factor,
            )
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=training_configs.learning_rate
            ),
            weight_exposure_loss=training_configs.weight_exposure_loss,
            weight_color_constancy_loss=training_configs.weight_color_constancy_loss,
            weight_illumination_smoothness_loss=training_configs.weight_illumination_smoothness_loss,
        )

    callbacks = [
        get_model_checkpoint_callback(
            filepath="checkpoint",
            save_best_only=training_configs.save_best_checkpoint_only,
            using_wandb=True,
        )
    ]
    callbacks.append(WandbMetricsLogger(log_freq="batch"))

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=training_configs.epochs,
        callbacks=callbacks,
    )

    wandb.finish()


if __name__ == "__main__":
    app.run(main)
