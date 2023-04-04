import tensorflow as tf


class PSNRLoss(tf.keras.losses.Loss):
    """Implementation of Negative PSNR Loss defined as follows:
    
    $$\text { Loss }=-\sum_{i=1}^2 \operatorname{PSNR}\left(\left(R_i+X_i\right), Y\right)$$

    References:

    1. [HINet: Half Instance Normalization Network for Image Restoration](https://arxiv.org/abs/2105.06086)
    """

    def __init__(self, max_val: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_val = max_val

    def call(self, y_true, y_pred):
        return -tf.image.psnr(y_true, y_pred, max_val=self.max_val)
