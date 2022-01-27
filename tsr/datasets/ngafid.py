import gdown
import pandas as pd
import numpy as np
from sklearn import preprocessing
from tqdm.autonotebook import tqdm
import tensorflow as tf
from loguru import logger

from tsr.utils import shell_exec
from tsr.datasets.common import fix_type, Reshaper
from tsr.methods.augmentation.random_shift import RandomShifter
from tsr.config import Config


class NGAFID_DatasetManager:

    ngafid_urls = {
        "2021_IAAI_C28": "https://drive.google.com/uc?id=1R5q2s-QavuI6DKj9z2rNxQPIOrbJlwUM",
        "2021_IAAI_C37": "https://drive.google.com/uc?id=1RkEZnddzlwpAG5GCht0HBWBBIcvHfYlT",
    }

    input_columns = [
        "volt1",
        "volt2",
        "amp1",
        "amp2",
        "FQtyL",
        "FQtyR",
        "E1 FFlow",
        "E1 OilT",
        "E1 OilP",
        "E1 RPM",
        "E1 CHT1",
        "E1 CHT2",
        "E1 CHT3",
        "E1 CHT4",
        "E1 EGT1",
        "E1 EGT2",
        "E1 EGT3",
        "E1 EGT4",
        "OAT",
        "IAS",
        "VSpd",
        "NormAc",
        "AltMSL",
    ]

    def __init__(
        self,
        config: Config,
        name="2021_IAAI_C28",
        scaler=None,
    ):
        logger.info("Creating TF Datset for NGAFID Dataset %s" % name)
        self.config = config
        self.name = name
        self.scaler = scaler

        self.dataframe, self.dataframe_sources, self.scaler = self.get_ngafid_data_as_dataframe(
            name=name, scaler=scaler
        )

        self.create_folded_datasets()

    def prepare_tfdataset(self, ds, shuffle: bool = False, repeat: bool = False, aug: bool = False) -> tf.data.Dataset:
        logger.debug("Preparing basic TF Dataset for Training or Inference Usage")

        ds = ds.shuffle(self.config.hyperparameters.shuffle_buffer) if shuffle else ds
        ds = ds.repeat() if repeat else ds
        ds = ds.batch(self.config.hyperparameters.batch_size, drop_remainder=True)
        ds = ds.map(RandomShifter.from_config(self.config))

        if aug:
            logger.debug("Adding Augmentations when Preparing Dataset")
            pass
            # batch_aug = get_batch_aug()
            # ds = ds.map(batch_aug)

        # force shape
        desired_input_shape = [self.config.hyperparameters.batch_size] + list(self.config.model.input_shape)
        ds = ds.map(Reshaper(input_shape=desired_input_shape))

        ds = ds.map(lambda example: (example["input"], example["target"]))
        if self.config.hyperparameters.num_class > 2:
            ds = ds.map(lambda x, y: (x, tf.one_hot(y, self.config.hyperparameters.num_class)))
        else:
            ds = ds.map(lambda x, y: (x, tf.reshape(y, (self.config.hyperparameters.batch_size, 1))))

        logger.debug("Successfully prepared basic TF Dataset for Training or Inference Usage")
        return ds

    def create_folded_datasets(self):
        logger.debug("Creating Datasets, but using slices of the dataframe to get folds")

        self.folded_datasets = []
        df = self.dataframe
        for i in range(self.config.hyperparameters.NFOLD):
            self.folded_datasets.append(
                self.ngafid_dataframe_to_dataset(
                    df[df.split == i], truncate_last_timesteps=self.config.hyperparameters.truncate_last_timesteps
                )
            )

        logger.debug("Successfully created folded datasets")

    def get_train_and_val_for_fold(self, fold: int) -> (tf.data.Dataset, tf.data.Dataset):
        # TODO: Reorganize this method somehow since you will be using it in all datasets
        #       Actually just implement this as the abstractmethod and then super() it

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
    def get_ngafid_data_as_dataframe(
        cls, name: str, scaler: object = None, skip_scaler: bool = False
    ) -> (pd.DataFrame, pd.DataFrame, object):
        logger.debug("Downloading NGAFID Data")

        url = cls.ngafid_urls[name]
        output = "data.csv.gz"
        gdown.download(url, output, quiet=False)

        logger.debug("Unzipping Data")
        shell_exec("gzip -f -d data.csv.gz")

        filename = "data.csv"
        df_test = pd.read_csv(filename, nrows=100)

        float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
        float32_cols = {c: np.float16 for c in float_cols}

        logger.debug("Reading Full Dataframe")
        df = pd.read_csv(filename, engine="c", dtype=float32_cols)
        df["id"] = df.id.astype("int32")
        df = df.dropna()  # you can handle nans differently, but ymmv
        sources = df[["id", "plane_id", "split", "date_diff", "before_after"]].drop_duplicates()

        if not skip_scaler:
            df, scaler = cls.apply_scaler(df, scaler=scaler)

        logger.debug("Successfully Read Data as Dataframe")
        return df, sources, scaler

    @classmethod
    def apply_scaler(cls, df, scaler=None, apply=True) -> (pd.DataFrame, object):
        logger.debug("Applying Scaler")
        if scaler is None:
            logger.debug("Calculating New Scaler")
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(df.loc[:, cls.input_columns].sample(100000, random_state=0))

        if apply:
            logger.debug("Using Scaler to Transform Data")
            arr = df.loc[:, cls.input_columns].values
            res = scaler.transform(arr)

            for i, col in tqdm(enumerate(cls.input_columns)):
                df.loc[:, col] = res[:, i]

        logger.debug("Scaling Applied Successfully")
        return df, scaler

    @classmethod
    def ngafid_dataframe_to_dataset(cls, df=None, truncate_last_timesteps=4096) -> tf.data.Dataset:
        logger.debug("Converting Dataframe to Basic TF Dataset")
        ids = df.id.unique()

        sensor_datas = []
        afters = []

        logger.debug("Looping over each unique ID")
        for id in tqdm(ids):
            sensor_data = df[df.id == id].iloc[-truncate_last_timesteps:, :23].values

            sensor_data = np.pad(sensor_data, [[0, truncate_last_timesteps - len(sensor_data)], [0, 0]])

            sensor_data = tf.convert_to_tensor(sensor_data, dtype=tf.float32)

            after = df[df.id == id]["before_after"].iloc[0]

            sensor_datas.append(sensor_data)
            afters.append(after)

        logger.debug("Stacking lists of Tensors")
        sensor_datas = tf.stack(sensor_datas)
        afters = np.stack(afters)

        ds = tf.data.Dataset.from_tensor_slices((sensor_datas, afters))
        ds = ds.map(fix_type)
        ds = ds.map(lambda x, y: {"input": x, "target": y})

        logger.debug("Successfully Converted Dataframe to Basic TF Dataset")
        return ds
