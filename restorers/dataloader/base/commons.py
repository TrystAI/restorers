import tensorflow as tf


def random_horiontal_flip(low_image, enhanced_image):
    return tf.cond(
        tf.random.uniform(shape=(), maxval=1) < 0.5,
        lambda: (low_image, enhanced_image),
        lambda: (
            tf.image.flip_left_right(low_image),
            tf.image.flip_left_right(enhanced_image),
        ),
    )


def random_vertical_flip(low_image, enhanced_image):
    return tf.cond(
        tf.random.uniform(shape=(), maxval=1) < 0.5,
        lambda: (low_image, enhanced_image),
        lambda: (
            tf.image.flip_up_down(low_image),
            tf.image.flip_up_down(enhanced_image),
        ),
    )


def read_image(image_path: str, normalization_factor) -> tf.Tensor:
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, dtype=tf.float32) / normalization_factor
    return image
