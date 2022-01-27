import tensorflow as tf


class InceptionBlock(tf.keras.layers.Layer):
    def __init__(self, nb_filters=32, use_bottleneck=True, kernel_size=41, stride=1, depth=3):
        super().__init__()
        self.nb_filters = nb_filters
        self.use_bottleneck = use_bottleneck
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.bottleneck_size = 32
        self.stride = stride
        self.depth = depth

    def build(self, input_shape):
        input_tensor = tf.keras.Input(input_shape[1:])

        x = input_tensor
        shortcut_x = input_tensor

        for d in range(self.depth):
            stride = self.stride if d == self.depth - 1 else 1
            x = InceptionLayer(
                nb_filters=self.nb_filters,
                use_bottleneck=self.use_bottleneck,
                kernel_size=self.kernel_size,
                stride=stride,
            )(x)

        x = ShortcutLayer(stride=self.stride)([x, shortcut_x])

        self.model = tf.keras.models.Model(inputs=input_tensor, outputs=x)

    def call(self, x):
        return self.model(x)


class InceptionLayer(tf.keras.layers.Layer):
    def __init__(self, nb_filters=32, use_bottleneck=True, kernel_size=41, stride=1):

        super().__init__()
        self.nb_filters = nb_filters
        self.use_bottleneck = use_bottleneck
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.bottleneck_size = 32
        self.stride = stride

    def build(self, input_tensor):
        activation = "linear"
        stride = self.stride
        input_tensor = tf.keras.Input(input_tensor[1:])

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = tf.keras.layers.Conv1D(
                filters=self.bottleneck_size, kernel_size=1, padding="same", activation=activation, use_bias=False
            )(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(
                tf.keras.layers.Conv1D(
                    filters=self.nb_filters,
                    kernel_size=kernel_size_s[i],
                    strides=stride,
                    padding="same",
                    activation=activation,
                    use_bias=False,
                )(input_inception)
            )

        max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding="same")(input_tensor)

        conv_6 = tf.keras.layers.Conv1D(
            filters=self.nb_filters, kernel_size=1, padding="same", activation=activation, use_bias=False
        )(max_pool_1)

        conv_list.append(conv_6)

        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation="relu")(x)

        self.model = tf.keras.models.Model(inputs=input_tensor, outputs=x)

    def call(self, x):
        return self.model(x)


class ShortcutLayer(tf.keras.layers.Layer):
    def __init__(self, stride=1):
        super().__init__()
        self.stride = stride

    def build(self, input_shapes):
        input_tensor, shortcut_tensor = input_shapes
        input_tensor, shortcut_tensor = (tf.keras.Input(input_tensor[1:]), tf.keras.Input(shortcut_tensor[1:]))

        shortcut_y = tf.keras.layers.Conv1D(
            filters=int(input_tensor.shape[-1]), kernel_size=1, padding="same", use_bias=False
        )(shortcut_tensor)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)
        shortcut_y = tf.keras.layers.MaxPool1D(pool_size=self.stride, strides=self.stride, padding="same")(shortcut_y)

        x = tf.keras.layers.Add()([input_tensor, shortcut_y])
        x = tf.keras.layers.Activation("relu")(x)

        self.model = tf.keras.Model(inputs=(input_tensor, shortcut_tensor), outputs=x)

    def call(self, inputs):
        return self.model(inputs)
