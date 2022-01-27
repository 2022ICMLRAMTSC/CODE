import tensorflow as tf

from tsr.models.attention import EncoderLayer
from tsr.config import Config
from tsr.models.attention.positionalencoding import PositionalEncoding

class SimpleMHSA(tf.keras.Sequential):
    @classmethod
    def from_config(cls, config: Config):
        raise NotImplementedError

    def __init__(
        self,
        input_shape,
        num_class,
    ):
        d_model = 512
        num_heads = 8
        dff = 512

        super().__init__(
            ([
                tf.keras.layers.Input(input_shape),
                tf.keras.layers.Conv1D(d_model, 1, activation = 'tanh'),
                PositionalEncoding(d_model = d_model),
                EncoderLayer(d_model = d_model, num_heads = num_heads, dff = dff),
                EncoderLayer(d_model = d_model, num_heads = num_heads, dff = dff),
                tf.keras.layers.Lambda(lambda x: x[:, -1, :]),
                tf.keras.layers.Dense(num_class, activation = "softmax")
            ]
            )
        )
