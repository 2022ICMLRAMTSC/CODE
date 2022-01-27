from tsr.datasets.datasetmanager import DatasetManager
from tsr.config import Config
import gdown
import pandas as pd
import numpy as np
from sklearn import preprocessing
from tqdm.autonotebook import tqdm
import tensorflow as tf
from loguru import logger
import pyreadr as py
from sklearn.model_selection import KFold
from tsr.utils import shell_exec
from tsr.datasets.common import Reshaper
from compress_pickle import dump
import gc


class TEP_DatasetManager(DatasetManager):
    url = "https://drive.google.com/uc?id=1m6Gkp2tNnnlAzaAVLaWnC2TtXNX2wJV8"
    parquet_url = "https://drive.google.com/uc?id=1--5ItWe4axeZofryEKOwlITBuYFwOO-P"

    num_examples = 10500
    cache_name = "TEP_Cache.gz"
    dataframe_disk_name = "TEP_data.parquet"

    def __init__(self, config: Config):
        '''
        By default, downloads the parquet version of the TEP dataset. You can manually process the RData files
        if you want using the get_tep_data_as_dataframe method.

        Args:
            config:
        '''

        self.config = config
        self.dataframe, self.scaler = self.get_tep_data_as_dataframe()
        # self.dataframe = self.apply_scaler(self.dataframe, self.scaler)

        self.folded_datasets = self.get_split_train_dataset_from_dataframe(self.dataframe)
        # self.test_dataset = self.get_test_dataset_from_dataframe(self.dataframe)

    def prepare_tfdataset(self, ds, shuffle: bool = False, repeat: bool = False, aug: bool = False) -> tf.data.Dataset:
        logger.debug("Preparing basic TF Dataset for Training or Inference Usage")

        ds = ds.shuffle(self.config.hyperparameters.shuffle_buffer) if shuffle else ds
        ds = ds.repeat() if repeat else ds
        ds = ds.batch(self.config.hyperparameters.batch_size, drop_remainder=True)

        if aug:
            logger.debug("Adding Augmentations when Preparing Dataset")
            pass
            # batch_aug = get_batch_aug()
            # ds = ds.map(batch_aug)

        desired_input_shape = [self.config.hyperparameters.batch_size] + list(self.config.model.input_shape)
        ds = ds.map(Reshaper(input_shape=desired_input_shape))
        ds = ds.map(lambda example: (example["input"], example["target"]))
        if self.config.hyperparameters.num_class > 2:
            ds = ds.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), self.config.hyperparameters.num_class)))
        else:
            ds = ds.map(lambda x, y: (x, tf.reshape(y, (self.config.hyperparameters.batch_size, 1))))

        logger.debug("Successfully prepared basic TF Dataset for Training or Inference Usage")
        return ds

    def get_train_and_val_for_fold(self, fold: int):
        logger.debug("Retrieving Fold %i" % fold)
        config = self.config

        train = []
        for i in range(config.hyperparameters.NFOLD):
            if i == fold:
                val_ds = self.folded_datasets[i]
            else:
                train.append(self.folded_datasets[i])

        train_ds = None
        for ds in train:
            train_ds = ds if train_ds is None else train_ds.concatenate(ds)

        train_ds = self.prepare_tfdataset(train_ds, shuffle=True, repeat=True, aug=True)
        val_ds = self.prepare_tfdataset(val_ds, shuffle=False)

        logger.debug("Successfully Retrieved Fold %i" % fold)
        return train_ds, val_ds

    @classmethod
    def get_tep_data_as_dataframe(cls, process_raw_rdata=False):

        if process_raw_rdata:
            logger.warning('You need a lot of RAM to read the RDATA file. At least 32GB')

            output = "tep_dataset.zip"
            gdown.download(cls.url, output, quiet=False)
            logger.debug("Downloaded Data")

            shell_exec("unzip -q -n tep_dataset.zip")
            logger.debug("Unzipped Data")

            logger.debug("Reading Training Data")
            a1 = py.read_r("TEP_FaultFree_Training.RData")
            a2 = py.read_r("TEP_Faulty_Training.RData")
            logger.debug("Fixing Column Types")
            b1 = cls.fix_column_types(a1["fault_free_training"])
            b2 = cls.fix_column_types(a2["faulty_training"])

            a1 = None
            a2 = None
            gc.collect()

            logger.debug("Reading Testing Data")
            a3 = py.read_r("TEP_FaultFree_Testing.RData")
            a4 = py.read_r("TEP_Faulty_Testing.RData")
            logger.debug("Fixing Column Types")
            b3 = cls.fix_column_types(a3["fault_free_testing"])
            b4 = cls.fix_column_types(a4["faulty_testing"])

            b1["split"] = "train"
            b2["split"] = "train"
            b3["split"] = "test"
            b4["split"] = "test"

            logger.debug("Combining Data")
            df = pd.concat([b1, b2, b3, b4])

            df["id"] = df.faultNumber.apply(lambda x: int(x)) + df.simulationRun.apply(lambda x: int(x) * 100)

            scaler = preprocessing.MinMaxScaler()
            scaler.fit(df.iloc[:, 3:55][df.split == "train"].sample(1000000, random_state = 0))

            logger.debug("Writing Data to Disk")

            df.to_parquet(cls.dataframe_disk_name)
            dump(scaler, cls.cache_name)

            logger.debug("Retrieved Data as Dataframe")

        else:
            logger.info("Downloading Parquet")

            gdown.download(cls.parquet_url, cls.dataframe_disk_name, quiet=False)

            logger.debug("Reading Full Dataframe from Disk")

            df = pd.read_parquet(cls.dataframe_disk_name)

            logger.debug("Retrieved Data as Dataframe")

            logger.debug("Fitting Scaler")
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(df.iloc[:, 3:55][df.split == "train"].sample(1000000, random_state = 0))

        return df, scaler

    @classmethod
    def apply_scaler(cls, df, scaler):
        logger.debug('Applying Scaler')

        n = int(1e6)
        for i in tqdm(range(int(len(df) / n) + 1)):
            df.iloc[i * n:(i + 1) * n, 3:55] = scaler.transform(df.iloc[i * n:(i + 1) * n, 3:55])

        logger.debug('Applied Scaler')

        return df

    @staticmethod
    def get_train_dataset_from_dataframe(df):
        arr = df[df.split == "train"].iloc[:, :55].values
        arr = np.reshape(arr, (-1, 500, 55))
        ds = tf.data.Dataset.from_tensor_slices(arr)
        ds = ds.map(lambda x: {"input": x[:, 3:], "target": x[0, 0]})
        return ds

    @staticmethod
    def get_test_dataset_from_dataframe(df):
        arr = df[df.split == "test"].iloc[:, :55].values
        arr = np.reshape(arr, (-1, 960, 55))
        ds = tf.data.Dataset.from_tensor_slices(arr)
        ds = ds.map(lambda x: {"input": x[:, 3:], "target": x[0, 0]})
        return ds

    @staticmethod
    def fix_column_types(b1: pd.DataFrame):
        for col in b1.columns:
            b1.loc[:, col] = b1.loc[:, col].astype("float32")
        return b1

    def get_split_train_dataset_from_dataframe(self, df):

        arr = df[df.split == "train"].iloc[:, :55].values
        arr = np.reshape(arr, (-1, 500, 55))

        train_splits = []

        np.random.seed(42)
        indices = np.array([i for i in range(len(arr))])

        np.random.shuffle(indices)

        for train_split, val_split in KFold(5, shuffle=True, random_state=0).split(indices):

            val_split = indices[val_split]
            ds = tf.data.Dataset.from_tensor_slices(arr[val_split])
            ds = ds.map(lambda x: {"input": x[:, 3:], "target": x[0, 0]})
            train_splits.append(ds)

        return train_splits
