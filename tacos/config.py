from pixell import enmap, enplot, curvedsky, utils
import healpy as hp 
import camb
import numpy as np
import yaml

import pkgutil
from ast import literal_eval

from tacos import utils, data, models


class Config:

    # parses a config file's different sections and serves them as attributes
    # to clients, like mixing matrix or chain objects

    def __init(self, config_path, load_channels=True, load_components=True, verbose=True):
        try:
            config = utils.config_from_yaml_resource(config_path)
        except FileNotFoundError:
            config = utils.config_from_yaml_file(config_path)
