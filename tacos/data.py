#!/usr/bin/env python3

# classes and helper functions that support easy loading/packaging of data on-disk
# for analysis
# all maps have shape (num_splits, num_pol, ny, nx)
# all ivars have shape (num_splits, num_pol, num_pol, ny, nx)

import numpy as np
import yaml
import pkgutil

from soapack import interfaces as sints
from pixell import enmap
import healpy as hp

from tacos import utils, beam
from tacos.bandpass import BandPass

# this is the main class representing a singular band of data, ie from one instrument
# and one frequency (and optionally, one detector subset)
class Channel:
    """Channel instance holding map data, covariance data, bandpasses, and beams. 

    Parameters
    ----------
    instr : str
        The instrument of the data set to load. Must be one of "act", "planck", "wmap"
    band : str
        The band name within the instrument
    id : str, optional
        The subset of the instrument + band data, e.g. detectors, by default 'all'
    set : str, optional
        The data split, e.g. "set0", "set1", or "coadd"
    notes : str, optional
        Additional identifier to append to data filenames, by default None
    correlated_noise : bool, optional
        The noise model, by default False
    pysm : bool, optional
        Whether to load pysm data instead of actual data, by default False
    healpix : bool, optional
        Whether to use hp.read_map to load map data, by default False.
        Only possible if pysm is True. Works by seeking "healpix" as first word
        in filename notes.
    beam_kwargs : dict, optional
        kwargs to pass to beam.load_<instrument>_beam, by default None
    bandpass_kwargs : dict, optional
        kwargs to pass to BandPass.load_<instrument>_bandpass, by default None

    Raises
    ------
    ValueError
        Instrument must be one of "act", "planck", "wmap"
    """

    def __init__(self, instr, band, id=None, set=None, notes=None, correlated_noise=False, pysm=False, 
                    healpix=False, cmb=False, noise=False, beam_kwargs=None, bandpass_kwargs=None):
        
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
        self._pysm = pysm
        self._healpix = healpix

        # store data
        if self.correlated_noise:
            raise NotImplementedError('Correlated noise not yet implemented')
        else:
            covmat_type = 'icovar'

        # maps and icovars
        if pysm:
            map_instr = 'pysm'
            map_id = 'all'
            map_set = 'all'
            if healpix:
                if notes is None:
                    map_notes = 'healpix'
                else:
                    map_notes = 'healpix_' + notes
            else:
                map_notes = notes
        else:
            map_instr = instr
            map_id = id
            map_set = set
            map_notes = notes

        map_path = utils.data_dir_str('map', map_instr)
        map_path += utils.data_fn_str(type='map', instr=map_instr, band=band, id=map_id, set=map_set, notes=map_notes)
        
        if pysm and healpix:
            self._map = utils.atleast_nd(hp.read_map(map_path, field=None, dtype=np.float32), 3) # (nsplit, npol, npix)
        elif not healpix:
            self._map = utils.atleast_nd(enmap.read_map(map_path), 4) # (nsplit, npol, ny, nx)
        elif not pysm and healpix:
            raise NotImplementedError('There are no healpix maps of actual data')
        else:
            raise AssertionError("How did we end up here?")

        covmat_path = utils.data_dir_str('covmat', instr)
        covmat_path += utils.data_fn_str(type=covmat_type, instr=instr, band=band, id=id, set=set, notes=notes)
        self._covmat = utils.atleast_nd(enmap.read_map(covmat_path), 5) # (nsplit, npol, npol, ny, nx)

        # beams
        beam_path = utils.data_dir_str('beam', instr)
        beam_path += utils.data_fn_str(type='beam', instr=instr, band='all', id=id, set='all', notes=notes)
        if instr == 'act':
            self._beam = beam.load_act_beam(beam_path, band, **beam_kwargs)
        elif instr == 'planck':
            self._beam = beam.load_planck_beam(beam_path, band, **beam_kwargs)
        elif instr == 'wmap':
            self._beam = beam.load_wmap_beam(beam_path, band, **beam_kwargs)
        elif instr == 'pysm':
            pass

        # bandpasses
        bandpass_path = utils.data_dir_str('bandpass', instr)
        bandpass_path += utils.data_fn_str(type='bandpass', instr=instr, band='all', id=id, set='all', notes=notes)
        if pysm:
            self._bandpass = BandPass.load_pysm_bandpass(bandpass_path, instr, band, **bandpass_kwargs)
        else:
            if instr == 'act':
                self._bandpass = BandPass.load_act_bandpass(bandpass_path, band, **bandpass_kwargs)
            elif instr == 'planck':
                self._bandpass = BandPass.load_planck_bandpass(bandpass_path, band, **bandpass_kwargs)
            elif instr == 'wmap':
                self._bandpass = BandPass.load_wmap_bandpass(bandpass_path, band, **bandpass_kwargs)
            elif instr == 'pysm':
                pass
            else:
                raise ValueError(f'{instr} must be one of "act", "planck", "wmap"')

        # add realizations as necessary
        if cmb is not False:
            assert pysm, 'Can only add CMB realization to a simulated map'
            self._map += utils.get_cmb_sim(self._map.shape, self._map.wcs, dtype=self._map.dtype, seed=cmb)

        if noise is not False:
            assert pysm, 'Can only add a noise realization to a simulated map'
            if covmat_type == 'icovar':
                self._map += utils.get_icovar_noise_sim(icovar=self.covmat, seed=noise)
            else:
                raise NotImplementedError('Correlated noise not yet implemented')

    def convolve_to_beam(self, bell):
        pass

    def convolve_with_beam(self, bell):
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
    def pysm(self):
        return self._pysm

    @property
    def healpix(self):
        return self._healpix

    @property
    def map(self):
        return self._map

    @property
    def covmat(self):
        return self._covmat

    @property
    def beam(self):
        return self._beam

    @property
    def bandpass(self):
        return self._bandpass