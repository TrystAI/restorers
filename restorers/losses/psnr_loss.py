import tensorflow as tf


class PSNRLoss(tf.keras.losses.Loss):
    
    def __init__(self, max_val: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_val = max_val
    
    def call(self, y_true, y_pred):
        return -tf.image.psnr(y_true, y_pred, max_val=self.max_val)
