import numpy as np
import tensorflow as tf


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, maximum_position_encoding=1000, d_model=256):
        super(PositionalEncoding, self).__init__()

        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    def call(self, x):

        seq_len = tf.shape(x)[1]
        x += self.pos_encoding[:, :seq_len, :]

        return x
