from abc import ABC, abstractmethod
import tensorflow as tf

from tsr.config import Config


class DatasetManager(ABC):
    @abstractmethod
    def __init__(self, config: Config):
        """
        Should initialize via the config file.

        Args:
                config:
        """

    @abstractmethod
    def get_train_and_val_for_fold(self, fold: int) -> (tf.data.Dataset, tf.data.Dataset):
        """
        It should be able to return a particular fold for the dataset

        Args:
                fold:

        Returns:

        """
        pass
