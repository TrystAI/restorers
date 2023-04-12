import tensorflow as tf

from .utils import scale_tensor


class PSNRMetric(tf.keras.metrics.Metric):
    """Stateful Tensorflow metric for caclulating
    [Peak Signal-to-noise Ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio).

    Args:
        max_val (float): The dynamic range of the images
            (i.e., the difference between the maximum the and minimum allowed values).
    """

    def __init__(self, max_val: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_val = max_val
        self.psnr = tf.keras.metrics.Mean(name="psnr")

    def update_state(self, y_true, y_pred, *args, **kwargs):
        psnr = tf.image.psnr(
            scale_tensor(y_true), scale_tensor(y_pred), max_val=self.max_val
        )
        self.psnr.update_state(psnr, *args, **kwargs)

    def result(self):
        return self.psnr.result()

    def reset_state(self):
        self.psnr.reset_state()


class SSIMMetric(tf.keras.metrics.Metric):
    """Stateful Tensorflow metric for caclulating
    [Structural Similarity](https://en.wikipedia.org/wiki/Structural_similarity).

    Args:
        max_val (float): The dynamic range of the images
            (i.e., the difference between the maximum the and minimum allowed values).
    """

    def __init__(self, max_val: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_val = max_val
        self.ssim = tf.keras.metrics.Mean(name="ssim")

    def update_state(self, y_true, y_pred, *args, **kwargs):
        ssim = tf.image.ssim(
            scale_tensor(y_true), scale_tensor(y_pred), max_val=self.max_val
        )
        self.ssim.update_state(ssim, *args, **kwargs)

    def result(self):
        return self.ssim.result()

    def reset_state(self):
        self.ssim.reset_state()
