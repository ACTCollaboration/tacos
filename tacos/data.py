#!/usr/bin/env python3

# classes and helper functions that support easy loading/packaging of data on-disk
# for analysis
# all maps have shape (num_splits, num_pol, ny, nx)
# all ivars have shape (num_splits, num_pol, num_pol, ny, nx)

import numpy as np
import yaml

from soapack import interfaces as sints
from tacos import utils

import os
import pkg_resources

# copied from soapack.interfaces
def config_from_yaml(filename):
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config

# load config to data products
config = sints.dconfig['tacos']

# class dataset
# class that loads a "consistent" dataset, meaning pixelization always and possibly a common beam
# returns a dataset object, whose main purpose is to hold a dict of channel objects

# this is the main file format, which handles all inputs and outputs
ext_dict = {
    'map': 'fits',
    'icovar': 'fits',
    'bandpass': 'hdf5',
    'beam': 'hdf5',
    }

def data_str(type=None, instr=None, band=None, id=None, set=None, notes=''):
    data_str_template = '{type}_{instr}_{band}_{id}_{set}{notes}.{ext}'
    return data_str_template.format(
        type=type, instr=instr, band=band, id=id, set=set, notes=notes, ext=ext_dict[type]
        )

# this is the main class representing a singular band of data, ie from one instrument
# and one frequency (and optionally, one detector subset)
class Channel:

    def __init__(self, instrument=None, band=None, id=None, correlated_noise=False):
        
        self._instrument = instrument
        self._band = band
        if id is None:
            id = 'all'
        self._id = id
        self._correlated_noise = correlated_noise

        @property
        def instrument(self):
            return self._instrument

        @property
        def band(self):
            return self._band

        @property
        def id(self):
            return self._id 

        @property
        def correlated_noise(self):
            return self._correlated_noise

        
