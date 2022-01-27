from focal_loss import SparseCategoricalFocalLoss
import tensorflow as tf


class CategoricalFocalLoss(SparseCategoricalFocalLoss):
    """
    Categorical version of the focal loss
    """

    def call(self, y_true, y_pred):

        y_true = tf.argmax(y_true, axis=-1)
        return super().call(y_true, y_pred)
