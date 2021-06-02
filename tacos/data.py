#!/usr/bin/env python3

# classes and helper functions that support easy loading/packaging of data on-disk
# for analysis
# all maps have shape (num_splits, num_pol, ny, nx)
# all ivars have shape (num_splits, num_pol, num_pol, ny, nx)

import numpy as np
import yaml

from soapack import interfaces as sints
from pixell import enmap
from tacos import utils
from tacos.bandpass import BandPass

# copied from soapack.interfaces
def config_from_yaml(filename):
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config

# load config to data products
config = sints.dconfig['tacos']

# this is the main file format, which handles all inputs and outputs
ext_dict = {
    'map': 'fits',
    'icovar': 'fits',
    'bandpass': 'hdf5',
    'beam': 'hdf5',
    }

def data_str(type=None, instr=None, band=None, id=None, set=None, notes=None):
    if notes is None:
        notes = ''
    data_str_template = '{type}_{instr}_{band}_{id}_{set}{notes}.{ext}'
    return data_str_template.format(
        type=type, instr=instr, band=band, id=id, set=set, notes=notes, ext=ext_dict[type]
        )

# this is the main class representing a singular band of data, ie from one instrument
# and one frequency (and optionally, one detector subset)
class Channel:

    def __init__(self, instrument=None, band=None, id=None, correlated_noise=False, notes=None):
        
        # assign and store metadata
        self._instrument = instrument
        self._band = band
        if id is None:
            id = 'all'
        self._id = id
        self._correlated_noise = correlated_noise
        self._notes = notes

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
            if self._correlated_noise:
                raise NotImplementedError('Correlated noise not yet implemented')
            return self._correlated_noise

        @property
        def notes(self):
            return self._notes

        # get and store data
        if self.correlated_noise:
            pass
        else:
            covmat_type = 'icovar'
            set = 'coadd'

        map_path = config['maps_path'] + f'{instrument}/'
        map_path += data_str(type='map', instr=instrument, band=band, id=id, set=set, notes=notes)
        self._map = enmap.read_map(map_path)

        covmat_path = config['covmats_path'] + f'{instrument}/'
        covmat_path += data_str(type=covmat_type, instr=instrument, band=band, id=id, set=set, notes=notes)
        self._covmat = enmap.read_map(covmat_path)

        bandpass_path = config['bandpasses_path'] + f'{instrument}/'
        bandpass_path += data_str(type='bandpass', instr=instrument, band=band, id=id, set=set, notes=notes)

        

        
