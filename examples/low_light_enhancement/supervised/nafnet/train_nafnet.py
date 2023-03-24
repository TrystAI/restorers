import wandb
from wandb.keras import WandbMetricsLogger

from absl import app, flags, logging
from ml_collections.config_flags import config_flags

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from restorers.model import NAFNet
from restorers.dataloader import LOLDataLoader
from restorers.losses import CharbonnierLoss, PSNRLoss
from restorers.metrics import PSNRMetric, SSIMMetric
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


def main(_):
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
    if using_wandb:
        wandb.config.global_batch_size = batch_size

    data_loader = LOLDataLoader(
        image_size=data_loader_configs.image_size,
        bit_depth=data_loader_configs.bit_depth,
        val_split=data_loader_configs.val_split,
        visualize_on_wandb=data_loader_configs.visualize_on_wandb,
        dataset_artifact_address=data_loader_configs.dataset_artifact_address,
    )
    train_dataset, val_dataset = data_loader.get_datasets(batch_size=batch_size)
    logging.info("Created Tensorflow Datasets.")

    with strategy.scope():
        model = NAFNet(
            filters=model_configs.filters,
            middle_block_num=model_configs.middle_block_num,
            encoder_block_nums=model_configs.encoder_block_nums,
            decoder_block_nums=model_configs.decoder_block_nums,
        )
        loss = CharbonnierLoss(
            epsilon=training_configs.charbonnier_epsilon,
            reduction=tf.keras.losses.Reduction.SUM,
        )

        decay_steps = (
            len(data_loader.train_input_images) // batch_size
        ) * training_configs.epochs
        lr_schedule_fn = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=training_configs.initial_learning_rate,
            decay_steps=decay_steps,
            alpha=training_configs.minimum_learning_rate,
        )
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr_schedule_fn,
            weight_decay=training_configs.weight_decay,
            beta_1=training_configs.decay_rate_1,
            beta_2=training_configs.decay_rate_2,
        )
        logging.info(f"Using AdamW optimizer.")

        psnr_metric = PSNRMetric(max_val=training_configs.psnr_max_val)
        logging.info("Using Peak Signal-noise Ratio Metric.")
        ssim_metric = SSIMMetric(max_val=training_configs.ssim_max_val)
        logging.info("Using Structural Similarity Metric.")

        model.compile(
            optimizer=optimizer, loss=loss, metrics=[psnr_metric, ssim_metric]
        )

    callbacks = [
        get_model_checkpoint_callback(
            filepath="checkpoint", save_best_only=False, using_wandb=using_wandb
        )
    ]
    if using_wandb:
        callbacks.append(WandbMetricsLogger(log_freq="batch"))

    logging.info("Starting Training...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=training_configs.epochs,
        callbacks=callbacks,
    )
    logging.info("Training Completed.")

    if using_wandb:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
