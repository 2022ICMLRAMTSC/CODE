import tensorflow as tf
from tsr.config import Config

class SimpleRNN(tf.keras.Sequential):
    @classmethod
    def from_config(cls, config: Config):
        raise NotImplementedError

    def __init__(
        self,
        input_shape,
        num_class,
    ):

        super().__init__([tf.keras.layers.Input(input_shape),
                                 tf.keras.layers.LSTM(256, return_sequences=True),
                                 tf.keras.layers.LSTM(512, return_sequences=True),
                                 tf.keras.layers.LSTM(512),
                                 tf.keras.layers.Dense(num_class, activation='softmax'),
    ])