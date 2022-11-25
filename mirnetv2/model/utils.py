from typing import List, Union

import numpy as np
import tensorflow as tf


def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Source:
        https://github.com/huggingface/transformers/blob/main/src/transformers/tf_utils.py#L26-L46
    Args:
        tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.
    Returns:
        `List[int]`: The shape of the tensor as a list.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def match_dtype(x: tf.Tensor, y: tf.Tensor):
    """Utility to match data-types of two variables. Useful during mixed-
    precision training."""
    if x.dtype != y.dtype:
        y = tf.cast(y, x.dtype)
    return x, y
