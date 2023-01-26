from typing import List, Tuple

import tensorflow as tf
from absl import logging
from matplotlib import pyplot as plt
from PIL import Image
from wandb.keras import WandbModelCheckpoint


def initialize_device() -> tf.distribute.Strategy:
    devices = tf.config.list_physical_devices("GPU")
    if len(devices) > 1:
        device_names = ", ".join([device.name for device in devices])
        logging.info(f"Using Mirrored Strategy to train over {device_names}")
        return tf.distribute.MirroredStrategy()
    else:
        logging.info(f"Using One Device Strategy to train over {devices[0].name}")
        return tf.distribute.OneDeviceStrategy(device="GPU:0")


def get_model_checkpoint_callback(
    filepath: str, save_best_only: bool, using_wandb: bool
) -> tf.keras.callbacks.ModelCheckpoint:
    return (
        WandbModelCheckpoint(
            filepath=filepath,
            monitor="val_loss",
            save_best_only=save_best_only,
            save_weights_only=False,
            initial_value_threshold=None,
        )
        if using_wandb
        else tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor="val_loss",
            save_best_only=save_best_only,
            save_weights_only=False,
            initial_value_threshold=None,
        )
    )


def scale_tensor(tensor: tf.Tensor) -> tf.Tensor:
    """Utility for scaling the values of a tensor in [0,1]"""
    _min = tf.math.reduce_min(tensor)
    _max = tf.math.reduce_max(tensor)
    return (tensor - _min) / (_max - _min)


def plot_results(
    images: List[Image], titles: List[str], figure_size: Tuple[int, int] = (12, 12)
) -> None:
    """A simple utility for plotting the results"""
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()
