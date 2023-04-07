import tensorflow as tf


class CharbonnierLoss(tf.keras.losses.Loss):
    """The Charbonnier implemented as a `tf.keras.losses.Loss`.

    The Charbonnier loss, also known as the "smooth L1 loss," is a loss function that is used in
    image processing and computer vision tasks to balance the trade-off between the Mean Squared
    Error (MSE) and the Mean Absolute Error (MAE). It is defined as

    $$L=\\sqrt{\\left(\\left(x^{\\wedge} 2+\\varepsilon^{\\wedge} 2\\right)\\right)}$$

    where x is the error and ε is a small positive constant (typically on the order of 0.001). It
    is less sensitive to outliers than the mean squared error and less computationally expensive
    than the mean absolute error.

    Args:
        epsilon (float): a small positive constant.
    """

    def __init__(self, epsilon: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = tf.convert_to_tensor(epsilon)

    def call(self, y_true, y_pred):
        squared_difference = tf.square(y_true - y_pred)
        return tf.reduce_mean(tf.sqrt(squared_difference + tf.square(self.epsilon)))
