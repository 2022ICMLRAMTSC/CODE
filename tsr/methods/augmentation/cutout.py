from tsr.methods.augmentation import Augmentation
import tensorflow as tf
from typing import Union


class Cutout(Augmentation):
    def __init__(
        self,
        batch_size: int,
        do_prob: float,
        sequence_shape: Union[list, tuple],
        min_cutout_len: int,
        max_cutout_len: int,
        channel_drop_prob: float,
    ):
        """
        Linear Mix of two random MTS within the batch, for each MTS within the batch, with chance based on do_prob

        Args:
                min_cutout_len:
                max_cutout_len:
                channel_drop_prob:
                batch_size:
                do_prob:
                sequence_shape: in the form of [Length, Channels]
        """

        super().__init__()
        self.max_cutout_len = max_cutout_len
        self.min_cutout_len = min_cutout_len
        self.sequence_shape = sequence_shape
        self.batch_size = batch_size
        self.do_prob = do_prob
        self.channel_drop_prob = channel_drop_prob

    def call(self, example: dict) -> dict:

        x = example["input"]

        # apply the function across a tensor with shape [batchsize] to return
        # a tensor with shape [batchsize, length, channels]
        batch_cutout_masks = tf.map_fn(self.get_cutout_mask, tf.zeros((self.batch_size,)), dtype=tf.float32)
        x = x * batch_cutout_masks

        example["input"] = x

        return example

    def get_length_wise_cut_array(self):

        # generate an array representing timesteps, like [1, 2, 3...]
        time = tf.range(0, self.sequence_shape[0], dtype=tf.float32)

        # generate the start and end value
        start = tf.random.uniform((), maxval=self.sequence_shape[0] - self.max_cutout_len)
        end = start + tf.random.uniform((), minval=self.min_cutout_len, maxval=self.max_cutout_len)

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
        return tf.cast(tf.random.uniform((self.sequence_shape[1],)) < self.channel_drop_prob, tf.float32)

    def get_cutout_mask(self, nothing: tf.Tensor) -> tf.Tensor:
        """

        Args:
            nothing: this is just a placeholder

        Returns:
            the cutout mask with shape = self.sequence_shape
        """

        timesteps = tf.reshape(self.get_length_wise_cut_array(), (self.sequence_shape[0], 1))
        channels = tf.reshape(self.get_channel_wise_cut_array(), (1, self.sequence_shape[1]))

        # timesteps * channels returns 1s where we want the cutout to occur
        # the mask is in the inverse, where we want 1s to represent where the cutout does not occur
        cutout_mask = tf.cast(timesteps * channels < 0.99, tf.float32)
        return cutout_mask

    def singular_call(self, input: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError
