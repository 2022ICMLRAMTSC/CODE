from tsr.methods.augmentation import Augmentation
import tensorflow as tf
from typing import Union


class Cutmix(Augmentation):
    def __init__(
        self,
        batch_size: int,
        do_prob: float,
        sequence_shape: Union[list, tuple],
        min_cutmix_len: int,
        max_cutmix_len: int,
        channel_replace_prob: float,
    ):
        """
        For each MTS, select a section by location [length, channels] and replace it with another random MTS's same
        section by location.

        Args:
                min_cutmix_len:
                max_cutmix_len:
                channel_replace_prob:
                batch_size:
                do_prob:
                sequence_shape: in the form of [Length, Channels]
        """

        super().__init__()
        self.max_cutmix_len = max_cutmix_len
        self.min_cutmix_len = min_cutmix_len
        self.sequence_shape = sequence_shape
        self.batch_size = batch_size
        self.do_prob = do_prob
        self.channel_replace_prob = channel_replace_prob

    def call(self, example: dict) -> dict:
        x = example["input"]

        # apply the function across a tensor with shape [batchsize] to return
        # a tensor with shape [batchsize, length, channels]
        batch_cutmix_masks = tf.map_fn(self.get_cutmix_mask, tf.zeros((self.batch_size,)), dtype=tf.float32)

        # get a mixup addition sequence
        original_input = x
        mixup_addition = tf.random.shuffle(x)

        # return original sequence where cutmixmask == 1 and mixup sequence otherwise
        example["input"] = tf.where(batch_cutmix_masks == 1, original_input, mixup_addition)

        return example

    def get_length_wise_cut_array(self):
        # generate an array representing timesteps, like [1, 2, 3...]
        time = tf.range(0, self.sequence_shape[0], dtype=tf.float32)

        # generate the start and end value
        start = tf.random.uniform((), maxval=self.sequence_shape[0] - self.max_cutmix_len)
        end = start + tf.random.uniform((), minval=self.min_cutmix_len, maxval=self.max_cutmix_len)

        do = tf.cast(
            tf.random.uniform(
                (),
            )
            < self.do_prob,
            tf.float32,
        )

        # return 1 for values between start and end and 0 elsewhere
        return tf.cast(tf.logical_and(time > start, time < end), tf.float32) * do

    def get_channel_wise_cut_array(self):
        # generate an array representing which channels to cut
        return tf.cast(tf.random.uniform((self.sequence_shape[1],)) < self.channel_replace_prob, tf.float32)

    def get_cutmix_mask(self, nothing: tf.Tensor) -> tf.Tensor:
        """

        Args:
            nothing: this is just a placeholder

        Returns:
            the cutmix mask with shape = self.sequence_shape
        """

        timesteps = tf.reshape(self.get_length_wise_cut_array(), (self.sequence_shape[0], 1))
        channels = tf.reshape(self.get_channel_wise_cut_array(), (1, self.sequence_shape[1]))

        # timesteps * channels returns 1s where we want the cutmix to occur
        # the mask is in the inverse, where we want 1s to represent where the cutmix does not occur
        cutmix_mask = tf.cast(timesteps * channels < 0.99, tf.float32)
        return cutmix_mask

    def singular_call(self, input: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError
