#!/usr/bin/env python3

import numpy as np
import yaml

from importlib import resources

# class dataset
# class that loads a "consistent" dataset, meaning pixelization always and possibly a common beam
# returns a dataset object, whose main purpose is to hold a dict of channel objects

# class channel
# holds the data (maps, beams, bandpasses etc) for a particular intrument and frequency band combo
# has methods for getting the various products out

# copied from soapack.interfaces
def config_from_yaml(filename):
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config

channels_config = resources.files('tacos')

