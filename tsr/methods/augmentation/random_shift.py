from tsr.methods.augmentation import Augmentation
import tensorflow as tf
from typing import Union


class RandomShifter(Augmentation):
    def __init__(self, shift_backward_max: int, shift_forward_max: int, sequence_shape: Union[list, tuple]):
        """
        Padded random shift of a sequence. A sequence is shifted forward or back by n timesteps by choosing n as a
        value between the forward max and backward max. If backward max is provided as a negative number, it will
        be converted to positive.

        Args:
                shift_backward_max:
                shift_forward_max:
                sequence_shape:
        """

        super().__init__()
        self.shift_forward_max = shift_forward_max
        self.shift_backward_max = abs(shift_backward_max)
        self.sequence_shape = sequence_shape

    def call(self, example: dict) -> dict:
        input = example["input"]
        input = tf.map_fn(self.singular_call, input)
        example["input"] = input
        return example

    def singular_call(self, input: tf.Tensor) -> tf.Tensor:
        start = self.get_start_position()
        input = self.shift(input, start)
        return input

    def shift(self, input: tf.Tensor, start: tf.Tensor) -> tf.Tensor:

        input = tf.pad(input, [[self.shift_backward_max, self.shift_forward_max], [0, 0]])
        input = input[start : start + self.sequence_shape[0]]
        return input

    def get_start_position(self) -> tf.Tensor:
        start = tf.random.uniform(
            shape=[], minval=0, maxval=self.shift_backward_max + self.shift_forward_max, dtype=tf.int64
        )

        return start

    @classmethod
    def from_config(cls, config):

        return cls(
            shift_backward_max=config.augment.random_shift.backward,
            shift_forward_max=config.augment.random_shift.forward,
            sequence_shape=config.model.input_shape,
        )
