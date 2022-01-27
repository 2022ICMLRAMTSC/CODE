import tensorflow as tf

from tsr.models.attention import EncoderLayer
from tsr.config import Config


class ConvMHSA(tf.keras.Sequential):
    @classmethod
    def from_config(cls, config: Config):

        if config.hyperparameters.num_class > 2:
            output = tf.keras.layers.Dense(config.hyperparameters.num_class, activation="softmax")
        else:
            output = (tf.keras.layers.Dense(1, activation="sigmoid"),)

        mc = config.model
        return cls(
            input_shape=mc.input_shape,
            conv_filters=mc.conv_filters,
            kernel=mc.kernel,
            strides=mc.strides,
            num_heads=mc.num_heads,
            d_model=mc.d_model,
            dff=mc.dff,
            output=output,
        )

    def __init__(
        self,
        input_shape=(4096, 23),
        conv_filters=[128, 128, 256, 256, 512],
        kernel=[7, 7, 3, 7, 7],
        strides=[1, 2, 1, 2, 2],
        num_heads=8,
        d_model=512,
        dff=512,
        output: tf.keras.layers.Layer = tf.keras.layers.Dense(1, activation="sigmoid"),
    ):

        super().__init__(
            [
                tf.keras.Input(shape=input_shape),
            ]
            + [
                tf.keras.layers.Conv1D(
                    filters=conv_filters[i],
                    kernel_size=kernel[i],
                    strides=strides[i],
                    padding="same",
                    activation="relu",
                )
                for i in range(len(kernel))
            ]
            + [
                EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff),
                EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff),
                EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff),
                EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff),
                tf.keras.layers.GlobalAveragePooling1D(),
                output,
            ]
        )

class ConvMHSA2(tf.keras.Sequential):
    @classmethod
    def from_config(cls, config: Config):
        raise NotImplementedError

    def __init__(
        self,
        input_shape,
        num_class,
    ):

        conv_filters= [128, 128, 256, 256, 512]
        kernel=[7, 7, 3, 7, 7]
        strides=[1, 2, 1, 2, 2]
        d_model = 512
        num_heads = 8
        dff = 512
        super().__init__(
            [
                tf.keras.Input(shape=input_shape),
            ]
            + [
                tf.keras.layers.Conv1D(
                    filters=conv_filters[i],
                    kernel_size=kernel[i],
                    strides=strides[i],
                    padding="same",
                    activation="relu",
                )
                for i in range(len(kernel))
            ]
            + [
                EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff),
                EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff),
                EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff),
                EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(num_class, activation = "softmax")
            ]
        )

