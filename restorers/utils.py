from typing import List, Tuple

import numpy as np
import tensorflow as tf
from absl import logging
from matplotlib import pyplot as plt
from PIL import Image

import wandb
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
    images: List[Image.Image],
    titles: List[str],
    figure_size: Tuple[int, int] = (12, 12),
) -> None:
    """A simple utility for plotting the results"""
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()


def fetch_wandb_artifact(artifact_address: str, artifact_type: str):
    return (
        wandb.Api().artifact(artifact_address, type=artifact_type).download()
        if wandb.run is None
        else wandb.use_artifact(artifact_address, type=artifact_type).download()
    )


def count_params(weights) -> int:
    """Count the total number of scalars composing the weights.
    Reference:
        https://github.com/keras-team/keras/blob/e6784e4302c7b8cd116b74a784f4b78d60e83c26/keras/utils/layer_utils.py#L107
    Args:
        weights: An iterable containing the weights on which to compute params
    Returns:
        (int): The total number of scalars composing the weights
    """
    unique_weights = {id(w): w for w in weights}.values()
    # Ignore TrackableWeightHandlers, which will not have a shape defined.
    unique_weights = [w for w in unique_weights if hasattr(w, "shape")]
    weight_shapes = [w.shape.as_list() for w in unique_weights]
    standardized_weight_shapes = [
        [0 if w_i is None else w_i for w_i in w] for w in weight_shapes
    ]
    return int(sum(np.prod(p) for p in standardized_weight_shapes))


def calculate_gflops(model, input_shape: List[int]) -> float:
    if not isinstance(model, (tf.keras.models.Sequential, tf.keras.models.Model)):
        raise ValueError(
            "Calculating FLOPS is only supported for "
            "`tf.keras.Model` and `tf.keras.Sequential` instances."
        )

    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    # Compute FLOPs for one sample
    inputs = [tf.TensorSpec([1] + input_shape, tf.float32)]

    # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPs with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
        .with_empty_output()
        .build()
    )

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    # convert to GFLOPs
    return (flops.total_float_ops / 1e9) / 2
