from __future__ import annotations

from collections import UserDict
from yaml import load, FullLoader
from schema import Schema, And, Use, Optional
import os
import inspect
from loguru import logger


class Config(UserDict):

    schema = Schema(
        {
            "hyperparameters": {
                "NFOLD": And(Use(int), int),
                "truncate_last_timesteps": And(Use(int), int),
                "num_class": And(Use(int), int),
                "batch_size": And(Use(int), int),
                "training_epochs": And(Use(int), int),
                "training_steps_per_epoch": And(Use(int), int),
                "shuffle_buffer": And(Use(int), int),
            },
            "model": {
                "architecture_name": And(Use(str), str),
                "input_shape": And(Use(list), list, lambda x: len(x) == 2, lambda x: all([type(v) is int for v in x])),
                Optional(object): object,
            },
            "augment": {
                "random_shift": {
                    "backward": And(Use(int), int),
                    "forward": And(Use(int), int),
                }
            },
            "environment": {
                "type": And(Use(str), str),
            },
            Optional(object): object,  # for allowing all keys, should be removed at some point probably
        }
    )

    configpypath = inspect.getfile(inspect.currentframe())

    def __init__(self, path_to_config="config.yaml"):
        """
        The config object is used as a dictionary with added functionality to pass along parameters
        for training and evaluation. These files are validated via the validate_yaml function.

        Args:
            path_to_config: path to a config file
        """

        self.yaml_contents = load(open(path_to_config, "r").read(), Loader=FullLoader)
        self.path_to_config = path_to_config
        self.config_dir = os.path.dirname(path_to_config)

        # Skip validation for now
        # self.validate_yaml()
        super(Config, self).__init__(self.yaml_contents)
        self.collect_sub_configs()

        logger.debug(self.data)

    def validate_yaml(self):
        """
        This method performs the assertions in the try block to validate the yaml file.

        Returns:
            Raises an error or returns nothing
        """

        self.yaml_contents = self.schema.validate(self.yaml_contents)

    def as_attr_dict(self, check=True):
        """
        Allows access of values as attributes, making code more readable and/or easier to type.

        Eg. config.task.num_classes vs config['task']['num_classes']

        Returns:
            None
        """

        attrdict = AttrDict.from_nested_dicts(self)

        for k in self.keys():
            if check:
                assert not hasattr(self.__class__, k), (
                    "A config top level section has the same name as an existing attribute of the config class. See %s"
                    % k
                )
            setattr(self, k, attrdict[k])

    @classmethod
    def get_default_config(cls):
        """
        We expect the default config to be stored alongside the config.py

        TODO: Deprecate this you really shouldn't have it

        Returns:
            a config object initialized from the default config file
        """

        path = os.path.join(os.path.dirname(cls.configpypath), "config.yaml")

        return cls(path_to_config=path)

    @classmethod
    def get_standard_config(cls, config_name):
        """
        Standard configs are stored in the configs folder in the package

        Args:
            config_name:

        Returns:
            config object initialized from a standard config file
        """

        path = os.path.join(os.path.dirname(cls.configpypath), "configs", config_name)
        return cls(path_to_config=path)

    def merge_config(self, config: Config):
        self.update(config)

    def collect_sub_configs(self):
        for k, v in self.items():

            try:
                sub_config_path = self[k]["sub_config_path"]
                sub_config_path = os.path.join(self.config_dir, sub_config_path)
                sub_config = Config(sub_config_path)

                self.merge_config(sub_config)
                logger.info("Added Subconfig for %s from %s" % (k, sub_config_path))

            except KeyError:
                logger.debug("Sub Config Not Specified for %s config group" % k)

    # def update(self, *args, **kwargs):
    #     super().update(*args, **kwargs)
    #     self.as_attr_dict()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        """Dictionary subclass whose entries can be accessed by attributes (as well
        as normally).
        """
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def from_nested_dicts(cls, data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict) and not isinstance(data, Config):
            return data
        else:
            return cls({key: cls.from_nested_dicts(data[key]) for key in data})


if __name__ == "__main__":
    # c = Config()
    c = Config.get_standard_config("TEP.yaml")

    # c.model = 'abc'
    # c.__dict__['model'] = 'abc'
    # c['model'] = 'abc'

    # print(c.model)
    # print(c['model'])
