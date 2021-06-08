#!/usr/bin/env python3

# classes and helper functions that support easy loading/packaging of data on-disk
# for analysis
# all maps have shape (num_splits, num_pol, ny, nx)
# all ivars have shape (num_splits, num_pol, num_pol, ny, nx)

import numpy as np
import yaml

from soapack import interfaces as sints
from pixell import enmap
from tacos import utils, beam
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
    instr : str
        The instrument of the data set to load. Must be one of "act", "planck", "wmap", or "pysm"
    band : str
        The band name within the instrument
    id : str, optional
        The subset of the instrument + band data, e.g. detectors, by default 'all'
    set : str, optional
        The data split, e.g. "set0", "set1", or "coadd
    notes : str, optional
        Additional identifier to append to data filenames, by default None
    correlated_noise : bool, optional
        The noise model, by default False
    beam_kwargs : dict, optional
        kwargs to pass to beam.load_<instrument>_beam, by default None
    bandpass_kwargs : dict, optional
        kwargs to pass to BandPass.load_<instrument>_bandpass, by default None

    Raises
    ------
    ValueError
        Instrument must be one of "act", "planck", "wmap", or "pysm"
    """

    def __init__(self, instr, band, id=None, set=None, notes=None, correlated_noise=False, 
                    beam_kwargs=None, bandpass_kwargs=None):
        
        # modify args/kwargs
        if beam_kwargs is None:
            beam_kwargs = {}
        if bandpass_kwargs is None:
            bandpass_kwargs = {}

        # store metadata
        self._instr = instr
        self._band = band
        if id is None:
            id = 'all'
        self._id = id
        if set is None:
            set = 'coadd'
        self._set = set
        self._notes = notes
        self._correlated_noise = correlated_noise

        # store data
        if self.correlated_noise:
            pass
        else:
            covmat_type = 'icovar'

        # maps and icovars
        map_path = config['maps_path'] + f'{instr}/'
        map_path += data_str(type='map', instr=instr, band=band, id=id, set=set, notes=notes)
        self._map = utils.atleast_nd(enmap.read_map(map_path), 4) # (nsplit, npol, ny, nx)

        covmat_path = config['covmats_path'] + f'{instr}/'
        covmat_path += data_str(type=covmat_type, instr=instr, band=band, id=id, set=set, notes=notes)
        self._covmat = utils.atleast_nd(enmap.read_map(covmat_path), 5) # (nsplit, npol, npol, ny, nx)

        # beams
        beam_path = config['beams_path'] + f'{instr}/'
        beam_path += data_str(type='beam', instr=instr, band='all', id=id, set='all', notes=notes)
        self.beam = beam.load_planck_beam(beam_path, band, **beam_kwargs)

        # bandpasses
        bandpass_path = config['bandpasses_path'] + f'{instr}/'
        bandpass_path += data_str(type='bandpass', instr=instr, band='all', id=id, set='all', notes=notes)
        if instr == 'act':
            self._bandpass = BandPass.load_act_bandpass(bandpass_path, band, **bandpass_kwargs)
        elif instr == 'planck':
            self._bandpass = BandPass.load_planck_bandpass(bandpass_path, band, **bandpass_kwargs)
        elif instr == 'wmap':
            self._bandpass = BandPass.load_wmap_bandpass(bandpass_path, band, **bandpass_kwargs)
        elif instr == 'pysm':
            pass
        else:
            raise ValueError(f'{instr} must be one of "act", "planck", "wmap", or "pysm"')

    def convolve_to_beam(self, beam):
        pass

    def convolve_with_beam(self, beam):
        pass

    @property
    def instr(self):
        return self._instr

    @property
    def band(self):
        return self._band

    @property
    def id(self):
        return self._id 

    @property
    def set(self):
        return self._set 

    @property
    def notes(self):
        return self._notes

    @property
    def correlated_noise(self):
        if self._correlated_noise:
            raise NotImplementedError('Correlated noise not yet implemented')
        return self._correlated_noise

    @property
    def map(self):
        return self._map

    @property
    def covmat(self):
        return self._covmat

    @property
    def bandpass(self):
        return self._bandpass