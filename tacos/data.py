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
    """Returns a generic data filename, of format '{type}_{instr}_{band}_{id}_{set}{notes}.{ext}'
    """
    if notes is None:
        notes = ''
    data_str_template = '{type}_{instr}_{band}_{id}_{set}{notes}.{ext}'
    return data_str_template.format(
        type=type, instr=instr, band=band, id=id, set=set, notes=notes, ext=ext_dict[type]
        )

# this is the main class representing a singular band of data, ie from one instrument
# and one frequency (and optionally, one detector subset)
class Channel:
    """Channel instance holding map data, covariance data, bandpasses, and beams. 

    Parameters
    ----------
    instrument : str
        The instrument of the data set to load. Must be one of "act", "planck", "wmap", or "pysm"
    band : str
        The band name within the instrument
    id : str, optional
        The subset of the instrument + band data, e.g. detectors, by default 'all'
    correlated_noise : bool, optional
        The noise model, by default False
    notes : str, optional
        Additional identifier to append to data filenames, by default None
    bandpass_kwargs : dict, optional
        kwargs to pass to BandPass.load_<instrument>_bandpass, by default None

    Raises
    ------
    ValueError
        Instrument must be one of "act", "planck", "wmap", or "pysm"
    """

    def __init__(self, instrument, band, id=None, correlated_noise=False, notes=None,
                    bandpass_kwargs=None):
        
        # modify args/kwargs
        if bandpass_kwargs is None:
            bandpass_kwargs = {}

        # store metadata
        self._instrument = instrument
        self._band = band
        if id is None:
            id = 'all'
        self._id = id
        self._correlated_noise = correlated_noise
        self._notes = notes

        # store data
        if self.correlated_noise:
            pass
        else:
            covmat_type = 'icovar'
            set = 'coadd'

        # maps and icovars
        map_path = config['maps_path'] + f'{instrument}/'
        map_path += data_str(type='map', instr=instrument, band=band, id=id, set=set, notes=notes)
        self._map = enmap.read_map(map_path)

        covmat_path = config['covmats_path'] + f'{instrument}/'
        covmat_path += data_str(type=covmat_type, instr=instrument, band=band, id=id, set=set, notes=notes)
        self._covmat = enmap.read_map(covmat_path)

        # bandpasses
        # filenames vary by instrument
        bandpass_path = config['bandpasses_path'] + f'{instrument}/'
        if instrument == 'act':
            bandpass_path += data_str(type='bandpass', instr=instrument, band='all', id=id, set='all', notes=notes)
            self._bandpass = BandPass.load_act_bandpass(bandpass_path, band, **bandpass_kwargs)
        elif instrument == 'planck':
            bandpass_path += data_str(type='bandpass', instr='hfi', band='all', id=id, set='all', notes=notes)
            self._bandpass = BandPass.load_hfi_bandpass(bandpass_path, band, **bandpass_kwargs)
        elif instrument == 'wmap':
            bandpass_path += data_str(type='bandpass', instr=instrument, band='all', id=id, set='all', notes=notes)
            self._bandpass = BandPass.load_wmap_bandpass(bandpass_path, band, **bandpass_kwargs)
        elif instrument == 'pysm':
            pass
        else:
            raise ValueError(f'{instrument} must be one of "act", "planck", "wmap", or "pysm"')

    def convolve_to_beam(self, beam):
        pass

    def convolve_with_beam(self, beam):
        pass

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

    @property
    def map(self):
        return self._map

    @property
    def covmat(self):
        return self._covmat

    @property
    def bandpass(self):
        return self._bandpass