import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Union


def fix_type(x, y):
    return tf.cast(x, tf.float32), tf.cast(y, tf.float32)


class Transform(ABC):
    @abstractmethod
    def __init__(self):
        """
        Transforms should be defined as a callable class with an initialization function specifying
        the parameters for the callable. For example, a resize transform would specify the target
        X and Y shapes, if it is for an image.

        Note that the callable function should always accept a dictionary, as that is the preferred unit
        for tf datasets.

        Augmentations should operate on batches, rather than single examples.
        """

    @abstractmethod
    def call(self, example: dict) -> dict:
        """
        This is the batch wise call of the function.

        Args:
                example: a batched time series

        Returns:
                dict
        """

    @abstractmethod
    def singular_call(self, input: tf.Tensor) -> tf.Tensor:
        """
        This is a call for a single tensor, not batched. Not all augmentations can execute a singular call.
        Where it cannot, simple pass on the singular call.

        Args:
                input: tf.Tensor

        Returns:
                tf.Tensor
        """

    def __call__(self, example: dict):
        return self.call(example)


class Reshaper(Transform):
    def __init__(self, input_shape: Union[list, tuple, None] = None, target_shape: Union[list, tuple, None] = None):
        self.input_shape = input_shape
        self.target_shape = target_shape

    def singular_call(self):
        raise NotImplementedError("This is purely a batchwise operation")

    def call(self, example: dict) -> dict:
        """



        Args:
            example:

        Returns:

        """

        if self.input_shape is not None:
            example["input"] = tf.reshape(example["input"], self.input_shape)
        if self.target_shape is not None:
            example["target"] = tf.reshape(example["target"], self.target_shape)

        return example
