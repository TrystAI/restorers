import tensorflow as tf


class PSNRLoss(tf.keras.losses.Loss):
    """Implementation of Negative PSNR Loss
    
    References:
    
    1. [HINet: Half Instance Normalization Network for Image Restoration](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Chen_HINet_Half_Instance_Normalization_Network_for_Image_Restoration_CVPRW_2021_paper.pdf)
    """
    
    def __init__(self, max_val: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_val = max_val
    
    def call(self, y_true, y_pred):
        return -tf.image.psnr(y_true, y_pred, max_val=self.max_val)
