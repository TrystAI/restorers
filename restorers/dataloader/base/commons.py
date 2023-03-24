import random
from typing import Sequence

import tensorflow as tf


def random_horiontal_flip(low_image, enhanced_image) -> Sequence[tf.Tensor]:
    seed = random.randint(0, 1000)
    low_image = tf.image.random_flip_left_right(low_image, seed=seed)
    enhanced_image = tf.image.random_flip_left_right(low_image, seed=seed)
    return low_image, enhanced_image


def random_unpaired_horiontal_flip(low_image) -> tf.Tensor:
    seed = random.randint(0, 1000)
    low_image = tf.image.random_flip_left_right(low_image, seed=seed)
    return low_image


def random_vertical_flip(low_image, enhanced_image) -> Sequence[tf.Tensor]:
    seed = random.randint(0, 1000)
    low_image = tf.image.random_flip_up_down(low_image, seed=seed)
    enhanced_image = tf.image.random_flip_up_down(low_image, seed=seed)
    return low_image, enhanced_image


def random_unpaired_vertical_flip(low_image) -> tf.Tensor:
    seed = random.randint(0, 1000)
    low_image = tf.image.random_flip_up_down(low_image, seed=seed)
    return low_image


def read_image(image_path: str, normalization_factor: float = 1.0) -> tf.Tensor:
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, dtype=tf.float32) / normalization_factor
    return image
