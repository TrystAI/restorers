import tensorflow as tf


def color_constancy(x):
    mean_rgb = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
    mean_red, mean_green, mean_blue = tf.split(mean_rgb, 3, axis=3)
    difference_red_green = tf.square(mean_red - mean_green)
    difference_red_blue = tf.square(mean_red - mean_blue)
    difference_green_blue = tf.square(mean_blue - mean_green)
    return tf.sqrt(
        tf.square(difference_red_green)
        + tf.square(difference_red_blue)
        + tf.square(difference_green_blue)
    )
