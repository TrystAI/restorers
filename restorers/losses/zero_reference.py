import tensorflow as tf


def color_constancy(x: tf.Tensor) -> tf.Tensor:
    """An implementation of the Color Constancy Loss.

    The purpose of the Color Constancy Loss is to correct the potential color deviations in the
    enhanced image and also build the relations among the three adjusted channels. It is given by

    $$L_{c o l}=\sum_{\forall(p, q) \in \varepsilon}\left(J^p-J^q\right)^2, \varepsilon=\{(R, G),(R, B),(G, B)\}$$

    Reference:

    1. [Zero-DCE: Zero-reference Deep Curve Estimation for Low-light Image Enhancement](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)
    2. [Zero-Reference Learning for Low-Light Image Enhancement (Supplementary Material)](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Guo_Zero-Reference_Deep_Curve_CVPR_2020_supplemental.pdf)
    3. [Official PyTorch implementation of Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L9)
    4. [Tensorflow implementation of Zero-DCE](https://github.com/tuvovan/Zero_DCE_TF/blob/master/src/loss.py#L10)
    5. [Keras tutorial for implementing Zero-DCE](https://keras.io/examples/vision/zero_dce/#color-constancy-loss)

    Args:
        x (tf.Tensor): image.
    """
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


def exposure_control_loss(
    x: tf.Tensor, window_size: int = 16, mean_val: float = 0.6
) -> tf.Tensor:
    """An implementation of the Exposure Constancy Loss.

    The exposure control loss measures the distance between the average intensity value of a local
    region to the well-exposedness level E which is set within [0.4, 0.7]. It is given by

    $$L_{e x p}=\frac{1}{M} \sum_{k=1}^M\left|Y_k-E\right|$$

    Reference:

    1. [Zero-DCE: Zero-reference Deep Curve Estimation for Low-light Image Enhancement](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)
    2. [Zero-Reference Learning for Low-Light Image Enhancement (Supplementary Material)](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Guo_Zero-Reference_Deep_Curve_CVPR_2020_supplemental.pdf)
    3. [Official PyTorch implementation of Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L74)
    4. [Tensorflow implementation of Zero-DCE](https://github.com/tuvovan/Zero_DCE_TF/blob/master/src/loss.py#L21)
    5. [Keras tutorial for implementing Zero-DCE](https://keras.io/examples/vision/zero_dce/#exposure-loss)

    Args:
        x (tf.Tensor): image.
        window_size (int): The size of the window for each dimension of the input tensor for average pooling.
        mean_val (int): The average intensity value of a local region to the well-exposedness level.
    """
    x = tf.reduce_mean(x, axis=-1, keepdims=True)
    mean = tf.nn.avg_pool2d(x, ksize=window_size, strides=window_size, padding="VALID")
    return tf.reduce_mean(tf.square(mean - mean_val))


def illumination_smoothness_loss(x: tf.Tensor) -> tf.Tensor:
    """An implementation of the Illumination Smoothness Loss.

    The purpose of the illumination smoothness loss is to preserve the monotonicity relations between
    neighboring pixels and it is applied to each curve parameter map.

    Reference:

    1. [Zero-DCE: Zero-reference Deep Curve Estimation for Low-light Image Enhancement](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)
    2. [Zero-Reference Learning for Low-Light Image Enhancement (Supplementary Material)](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Guo_Zero-Reference_Deep_Curve_CVPR_2020_supplemental.pdf)
    3. [Official PyTorch implementation of Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L90)
    4. [Tensorflow implementation of Zero-DCE](https://github.com/tuvovan/Zero_DCE_TF/blob/master/src/loss.py#L28)
    5. [Keras tutorial for implementing Zero-DCE](https://keras.io/examples/vision/zero_dce/#illumination-smoothness-loss)

    Args:
        x (tf.Tensor): image.
    """
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
