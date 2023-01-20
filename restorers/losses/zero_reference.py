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


def exposure_control_loss(x, window_size: int = 16, mean_val: float = 0.6):
    x = tf.reduce_mean(x, axis=-1, keepdims=True)
    mean = tf.nn.avg_pool2d(x, ksize=window_size, strides=window_size, padding="VALID")
    return tf.reduce_mean(tf.square(mean - mean_val))


def illumination_smoothness_loss(x):
    batch_size = tf.shape(x)[0]
    h_x = tf.shape(x)[1]
    w_x = tf.shape(x)[2]
    count_h = (tf.shape(x)[2] - 1) * tf.shape(x)[3]
    count_w = tf.shape(x)[2] * (tf.shape(x)[3] - 1)
    h_tv = tf.reduce_sum(tf.square((x[:, 1:, :, :] - x[:, : h_x - 1, :, :])))
    w_tv = tf.reduce_sum(tf.square((x[:, :, 1:, :] - x[:, :, : w_x - 1, :])))
    batch_size = tf.cast(batch_size, dtype=tf.float32)
    count_h = tf.cast(count_h, dtype=tf.float32)
    count_w = tf.cast(count_w, dtype=tf.float32)
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
