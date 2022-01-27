from tsr.datasets.datasetmanager import DatasetManager
from tsr.config import Config
from tsr.utils import shell_exec
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.datatypes._panel._convert import (
    from_nested_to_3d_numpy,
)
from loguru import logger

import pandas as pd
import numpy as np
import tensorflow as tf


class UAE_DatasetManager(DatasetManager):

    url = "http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip"

    directories = {
        "ArticularyWordRecognition": {
            "TEST": "Multivariate_ts/ArticularyWordRecognition/ArticularyWordRecognition_TEST.ts",
            "TRAIN": "Multivariate_ts/ArticularyWordRecognition/ArticularyWordRecognition_TRAIN.ts",
        },
        "AtrialFibrillation": {
            "TEST": "Multivariate_ts/AtrialFibrillation/AtrialFibrillation_TEST.ts",
            "TRAIN": "Multivariate_ts/AtrialFibrillation/AtrialFibrillation_TRAIN.ts",
        },
        "BasicMotions": {
            "TEST": "Multivariate_ts/BasicMotions/BasicMotions_TEST.ts",
            "TRAIN": "Multivariate_ts/BasicMotions/BasicMotions_TRAIN.ts",
        },
        # "CharacterTrajectories": {
        #     "TEST": "Multivariate_ts/CharacterTrajectories/CharacterTrajectories_TEST.ts",
        #     "TRAIN": "Multivariate_ts/CharacterTrajectories/CharacterTrajectories_TRAIN.ts",
        # },
        "Cricket": {
            "TEST": "Multivariate_ts/Cricket/Cricket_TEST.ts",
            "TRAIN": "Multivariate_ts/Cricket/Cricket_TRAIN.ts",
        },
        "DuckDuckGeese": {
            "TEST": "Multivariate_ts/DuckDuckGeese/DuckDuckGeese_TEST.ts",
            "TRAIN": "Multivariate_ts/DuckDuckGeese/DuckDuckGeese_TRAIN.ts",
        },
        "ERing": {"TEST": "Multivariate_ts/ERing/ERing_TEST.ts", "TRAIN": "Multivariate_ts/ERing/ERing_TRAIN.ts"},
        "EigenWorms": {
            "TEST": "Multivariate_ts/EigenWorms/EigenWorms_TEST.ts",
            "TRAIN": "Multivariate_ts/EigenWorms/EigenWorms_TRAIN.ts",
        },
        "Epilepsy": {
            "TEST": "Multivariate_ts/Epilepsy/Epilepsy_TEST.ts",
            "TRAIN": "Multivariate_ts/Epilepsy/Epilepsy_TRAIN.ts",
        },
        "EthanolConcentration": {
            "TEST": "Multivariate_ts/EthanolConcentration/EthanolConcentration_TEST.ts",
            "TRAIN": "Multivariate_ts/EthanolConcentration/EthanolConcentration_TRAIN.ts",
        },
        "FaceDetection": {
            "TEST": "Multivariate_ts/FaceDetection/FaceDetection_TEST.ts",
            "TRAIN": "Multivariate_ts/FaceDetection/FaceDetection_TRAIN.ts",
        },
        "FingerMovements": {
            "TEST": "Multivariate_ts/FingerMovements/FingerMovements_TEST.ts",
            "TRAIN": "Multivariate_ts/FingerMovements/FingerMovements_TRAIN.ts",
        },
        "HandMovementDirection": {
            "TEST": "Multivariate_ts/HandMovementDirection/HandMovementDirection_TEST.ts",
            "TRAIN": "Multivariate_ts/HandMovementDirection/HandMovementDirection_TRAIN.ts",
        },
        "Handwriting": {
            "TEST": "Multivariate_ts/Handwriting/Handwriting_TEST.ts",
            "TRAIN": "Multivariate_ts/Handwriting/Handwriting_TRAIN.ts",
        },
        "Heartbeat": {
            "TEST": "Multivariate_ts/Heartbeat/Heartbeat_TEST.ts",
            "TRAIN": "Multivariate_ts/Heartbeat/Heartbeat_TRAIN.ts",
        },
        # "InsectWingbeat": {
        #     "TEST": "Multivariate_ts/InsectWingbeat/InsectWingbeat_TEST.ts",
        #     "TRAIN": "Multivariate_ts/InsectWingbeat/InsectWingbeat_TRAIN.ts",
        # },
        # "JapaneseVowels": {
        #     "TEST": "Multivariate_ts/JapaneseVowels/JapaneseVowels_TEST.ts",
        #     "TRAIN": "Multivariate_ts/JapaneseVowels/JapaneseVowels_TRAIN.ts",
        # },
        "LSST": {"TEST": "Multivariate_ts/LSST/LSST_TEST.ts", "TRAIN": "Multivariate_ts/LSST/LSST_TRAIN.ts"},
        "Libras": {"TEST": "Multivariate_ts/Libras/Libras_TEST.ts", "TRAIN": "Multivariate_ts/Libras/Libras_TRAIN.ts"},
        "MotorImagery": {
            "TEST": "Multivariate_ts/MotorImagery/MotorImagery_TEST.ts",
            "TRAIN": "Multivariate_ts/MotorImagery/MotorImagery_TRAIN.ts",
        },
        "NATOPS": {"TEST": "Multivariate_ts/NATOPS/NATOPS_TEST.ts", "TRAIN": "Multivariate_ts/NATOPS/NATOPS_TRAIN.ts"},
        "PEMS-SF": {
            "TEST": "Multivariate_ts/PEMS-SF/PEMS-SF_TEST.ts",
            "TRAIN": "Multivariate_ts/PEMS-SF/PEMS-SF_TRAIN.ts",
        },
        "PenDigits": {
            "TEST": "Multivariate_ts/PenDigits/PenDigits_TEST.ts",
            "TRAIN": "Multivariate_ts/PenDigits/PenDigits_TRAIN.ts",
        },
        "PhonemeSpectra": {
            "TEST": "Multivariate_ts/PhonemeSpectra/PhonemeSpectra_TEST.ts",
            "TRAIN": "Multivariate_ts/PhonemeSpectra/PhonemeSpectra_TRAIN.ts",
        },
        "RacketSports": {
            "TEST": "Multivariate_ts/RacketSports/RacketSports_TEST.ts",
            "TRAIN": "Multivariate_ts/RacketSports/RacketSports_TRAIN.ts",
        },
        "SelfRegulationSCP1": {
            "TEST": "Multivariate_ts/SelfRegulationSCP1/SelfRegulationSCP1_TEST.ts",
            "TRAIN": "Multivariate_ts/SelfRegulationSCP1/SelfRegulationSCP1_TRAIN.ts",
        },
        "SelfRegulationSCP2": {
            "TEST": "Multivariate_ts/SelfRegulationSCP2/SelfRegulationSCP2_TEST.ts",
            "TRAIN": "Multivariate_ts/SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.ts",
        },
        # "SpokenArabicDigits": {
        #     "TEST": "Multivariate_ts/SpokenArabicDigits/SpokenArabicDigits_TEST.ts",
        #     "TRAIN": "Multivariate_ts/SpokenArabicDigits/SpokenArabicDigits_TRAIN.ts",
        # },
        "StandWalkJump": {
            "TEST": "Multivariate_ts/StandWalkJump/StandWalkJump_TEST.ts",
            "TRAIN": "Multivariate_ts/StandWalkJump/StandWalkJump_TRAIN.ts",
        },
        "UWaveGestureLibrary": {
            "TEST": "Multivariate_ts/UWaveGestureLibrary/UWaveGestureLibrary_TEST.ts",
            "TRAIN": "Multivariate_ts/UWaveGestureLibrary/UWaveGestureLibrary_TRAIN.ts",
        },
    }

    def __init__(self, config: Config):
        self.download_and_unzip()

    def download_and_unzip(self):
        logger.info("Downloading UAE Archive for Multivariate TS Classification")
        shell_exec("wget http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip")
        shell_exec("unzip -q -n Multivariate2018_ts.zip")
        logger.debug("Unzipping completed")

    def get_dataset_as_sktime(self, dataset_name, split="TRAIN", format="TF"):
        path = self.directories[dataset_name][split]
        return load_from_tsfile_to_dataframe(path)

    def get_dataset_as_array(self, dataset_name, split="TRAIN"):
        path = self.directories[dataset_name][split]

        x, y = load_from_tsfile_to_dataframe(path)

        max_length = self.get_max_length(x)
        x = x.applymap(self.get_to_length(max_length))

        y = self.to_numeric_classes(y)
        x = self.convert_sktime_format_to_array(x)

        return x, y

    def get_datasets_as_tf(self, dataset_name, batch_size=64, shuffle=1000, scale=False, resample_seed = 0):
        """

        Args:
            dataset_name:
            batch_size:
            shuffle:
            resample_seed: if not 0, then resamples train and test so they each have the original number of each class,
            but examples are randomly distributed between train and test
        Returns:
            ds
            test_ds
            shape of the train dataset
            number of classes
        """
        x, y = self.get_dataset_as_array(dataset_name, split="TRAIN")
        x_test, y_test = self.get_dataset_as_array(dataset_name, split="TEST")

        x, y, x_test, y_test = self.resample_train_test(x, y, x_test, y_test, seed=resample_seed)


        if scale:
            minn = np.min(x)
            x = x - minn
            maxx = np.max(x)
            x = x / maxx
            x_test = x_test - minn
            x_test = x_test / maxx

        ds = tf.data.Dataset.from_tensor_slices((x, y))
        ds = ds.repeat()
        ds = ds.shuffle(shuffle) if shuffle else ds
        ds = ds.batch(64, drop_remainder=True)
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y))

        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(lambda x, y: (tf.cast(x, tf.float32), y))
        test_ds = test_ds.batch(1, drop_remainder=True)

        return ds, test_ds, x.shape, np.max(y) + 1

    @staticmethod
    def get_to_length(length):
        def to_length(a):
            return np.pad(a, (0, np.max(length - len(a), 0)))

        return to_length

    @staticmethod
    def get_max_length(x):
        return np.max(x.iloc[:, 0].apply(lambda x: len(x)))

    def get_train_and_val_for_fold(self, fold: int) -> (tf.data.Dataset, tf.data.Dataset):
        raise NotImplementedError

    @staticmethod
    def to_numeric_classes(y_train):
        s = pd.get_dummies(pd.Series(y_train))
        fixed_classes = np.argmax(s.values, axis=1)
        return fixed_classes

    @staticmethod
    def convert_sktime_format_to_array(x_train):
        """
        This should end up with (instances, length, channels)

        """
        arr = from_nested_to_3d_numpy(x_train)
        arr = np.transpose(arr, (0, 2, 1))
        return arr

    @staticmethod
    def resample_train_test(x, y, x_test, y_test, seed = 0):

        if not seed == 0:

            combined_y = np.concatenate([y_test, y])
            combined_x = np.concatenate([x_test, x])

            np.random.seed(seed = seed)
            train_indices = []
            test_indices = []

            for cls in range(np.max(combined_y) + 1):
                np.max(combined_y)

                arr = np.argwhere(combined_y == cls)

                np.random.shuffle(arr)

                train_index = arr[:len(np.argwhere(y == cls))]
                test_index = arr[len(np.argwhere(y == cls)):]
                train_indices.append(train_index)
                test_indices.append(test_index)

            train_indices = np.squeeze(np.concatenate(train_indices))
            test_indices = np.squeeze(np.concatenate(test_indices))

            x = combined_x[train_indices]
            y = combined_y[train_indices]
            x_test = combined_x[test_indices]
            y_test = combined_y[test_indices]

        return x, y, x_test, y_test
