from tsr.methods.augmentation import Augmentation
import tensorflow as tf
from typing import Union


class Mixup(Augmentation):
    def __init__(
        self,
        batch_size: int,
        do_prob: float,
        sequence_shape: Union[list, tuple],
        linear_mix_min: float = 0.1,
        linear_mix_max: float = 0.5,
    ):
        """
        Linear Mix of two random MTS within the batch, for each MTS within the batch, with chance based on do_prob

        Args:
                batch_size:
                do_prob:
                linear_mix_min:
                linear_mix_max:
                sequence_shape: in the form of [Length, Channels]
        """

        super().__init__()
        self.linear_mix_max = linear_mix_max
        self.linear_mix_min = linear_mix_min
        self.sequence_shape = sequence_shape
        self.batch_size = batch_size
        self.do_prob = do_prob

    def call(self, example: dict) -> dict:

        array_do_aug = tf.cast(
            tf.random.uniform((self.batch_size, 1, 1), minval=0.0, maxval=1.0) < self.do_prob, tf.float32
        )
        take_from_mixup_addition_percentage = tf.random.uniform(
            (), minval=self.linear_mix_min, maxval=self.linear_mix_max
        )

        take_from_mixup_addition_percentage = array_do_aug * take_from_mixup_addition_percentage

        x = example["input"]

        original_input = x
        mixup_addition = tf.random.shuffle(x)

        x = original_input * (1 - take_from_mixup_addition_percentage) + mixup_addition * (
            take_from_mixup_addition_percentage
        )

        example["input"] = x

        return example

    def singular_call(self, input: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError
