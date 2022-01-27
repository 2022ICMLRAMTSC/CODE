from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import Rocket
import numpy as np
import tensorflow as tf


class ROCKET:
    @classmethod
    def from_config(cls):
        raise NotImplementedError

    def __init__(self, num_kernels=20000):
        self.pipeline = make_pipeline(
            Rocket(num_kernels=num_kernels), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        )

    def fit(self, train_dataset, *args, **kwargs):

        assert issubclass(train_dataset, tf.data.Dataset)

        x = []
        y = []
        for example in train_dataset:
            x.append(example[0])
            y.append(example[1])

        self.pipeline.fit(x, y)

    def __call__(self):
        raise NotImplementedError

    def predict(self, dataset):
        self.pipeline.predict(dataset)
