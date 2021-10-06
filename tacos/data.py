#!/usr/bin/env python3

COADD_SPLIT_NUM = 103_094

# classes and helper functions that support easy loading/packaging of data on-disk
# for analysis
# all maps have shape (num_splits, num_pol, ny, nx)
# all ivars have shape (num_splits, num_pol, num_pol, ny, nx)

import numpy as np

from pixell import enmap
import healpy as hp

from tacos import utils, beam
from tacos.bandpass import BandPass

# this is the main class representing a singular band of data, ie from one instrument
# and one frequency (and optionally, one detector subset)
class Channel:

    def __init__(self, instr, band, id=None, set=None, notes=None, polstr=None, pysm_notes=None,
                    correlated_noise=False, pysm=False, healpix=False, cmb=False, cmb_kwargs=None,
                    sim_num=False, noise_kwargs=None, beam_kwargs=None, bandpass_kwargs=None, **kwargs):
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
            The data split, e.g. "set0", "set1". If None, passes "coadd"
        notes : str, optional
            Additional identifier to append to data filenames, by default None
        polstr : str, optional
            Which Stokes components to slice out of data and noise model (order preserved).
            If None, then retain all 3. Default is None.
        pysm_notes : str, optional
            Additional identifier unique to pysm data, by default None. Only operative if
            pysm is True. If notes is passed but pysm_notes is not, assume pysm_notes = notes
        correlated_noise : bool, optional
            The noise model, by default False
        pysm : bool, optional
            Whether to load pysm data instead of actual data, by default False
        healpix : bool, optional
            Whether to use hp.read_map to load map data, by default False.
            Only possible if pysm is True. Works by seeking "healpix" as first word
            in filename notes.
        cmb : bool, None, int, or tuple-of-int
            Whether to add a CMB realization to the map. False will pass; None, int, or
            tuple-of-int will be set as the seed of the realization. True will set cmb to
            None. Raises exception if pysm is False.
        cmb_kwargs : dict or None, optional
            Any kwargs to pass to utils.get_cmb_sim(...), by default None
        sim_num : None or int, optional
            Whether to add a noise realization to the map (None does not). int will help
            set the seed of the realization through NoiseModel.get_sim(sim_num=noise). 
            Raises exception if pysm is False. Default is None.
        noise_kwargs: dict or None, optional
            Any kwargs to pass to utils.get_icovar_noise_sim(...), by default None
        beam_kwargs : dict, optional
            kwargs to pass to beam.load_<instrument>_beam, by default None
        bandpass_kwargs : dict, optional
            kwargs to pass to BandPass.load_<instrument>_bandpass, by default None

        Raises
        ------
        ValueError
            Instrument must be one of "act", "planck", "wmap"
        """
        
        # modify args/kwargs
        if beam_kwargs is None:
            beam_kwargs = {}
        if bandpass_kwargs is None:
            bandpass_kwargs = {}

        # store metadata
        self.instr = instr
        self.band = band
        if id is None:
            id = 'all'
        self.id = id
        if set:
            self._split_num = int(set[-1]) # e.g. 'set3'
        else:
            set = 'coadd'
            self._split_num = COADD_SPLIT_NUM
        self.set = set
        self.notes = notes
        self._polidxs = utils.polstr2polidxs(polstr)
        self.correlated_noise = correlated_noise
        if self.correlated_noise:
            raise NotImplementedError('Correlated noise not yet implemented')
        self.pysm = pysm
        self.healpix = healpix

        # maps and icovars
        if pysm:
            map_instr = 'pysm'
            map_id = 'all'
            map_set = 'all'
            if not pysm_notes:
                pysm_notes = notes # may still be None if notes is None
            if healpix:
                if not pysm_notes:
                    map_notes = 'healpix'
                else:
                    map_notes = 'healpix_' + pysm_notes
            else:
                map_notes = pysm_notes
        else:
            map_instr = instr
            map_id = id
            map_set = set
            map_notes = notes

        map_path = utils.data_dir_str('map', map_instr)
        map_path += utils.data_fn_str(type='map', instr=map_instr, band=band, id=map_id, set=map_set, notes=map_notes)
        
        if pysm and healpix:
            self._map = hp.read_map(map_path, field=None, dtype=np.float32) # (npol, npix)
        elif not healpix:
            self._map = enmap.read_map(map_path) # (npol, ny, nx)
        elif not pysm and healpix:
            raise NotImplementedError('There are no healpix maps of actual data')
        else:
            raise AssertionError("How did we end up here?")
        self._map = self._map[self._polidxs]
        
        # add optional mult_fact from noise_kwargs
        noise_kwargs = {} if noise_kwargs is None else noise_kwargs
        mult_fact = noise_kwargs.get('mult_fact', 1)
        if self.correlated_noise:
            pass
        else:
            self._noise_model = SimplePixelNoiseModel(instr, band, id, set, notes=notes, polstr=polstr, mult_fact=mult_fact)

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

        # add cmb, noise realizations as necessary
        if cmb is not False:
            assert pysm, 'Can only add CMB realization to a simulated map'
            if cmb is True:
                cmb = None
            cmb_kwargs = {} if cmb_kwargs is None else cmb_kwargs
            self._map += utils.get_cmb_sim(self.map.shape, self.map.wcs, dtype=self.map.dtype, seed=cmb, **cmb_kwargs)

        if sim_num is not False:
            assert pysm, 'Can only add a noise realization to a simulated map'
            self._map += self._noise_model.get_sim(self._split_num, sim_num)

    def convolve_to_beam(self, bell):
        pass

    def convolve_with_beam(self, bell):
        pass

    @property
    def map(self):
        return self._map

    @property
    def noise_model(self):
        return self._noise_model

    @property
    def beam(self):
        return self._beam

    @property
    def bandpass(self):
        return self._bandpass

    
class SimplePixelNoiseModel:

    def __init__(self, instr, band, id, set, notes=None, polstr=None, mult_fact=1):
        self._instr = instr
        self._band = band
        self._id = id
        self._set = set
        self._notes = notes
        
        self._polidxs = utils.polstr2polidxs(polstr)

        self._mult_fact = mult_fact
        
        fn = self._get_model_fn(instr, band, set)
        self._nm_dict = self._read_model(fn)
        self._shape = self.model.shape
        self._dtype = self.model.dtype

    def _get_model_fn(self, instr, band, set):
        """Get a noise model filename for split split_num; return as <str>"""
        assert set in ['set0', 'set1', 'coadd']
        inv_cov_mat_path = utils.data_dir_str('covmat', instr)
        if not self._notes:
            notes = ''
        else:
            notes = '_' + self._notes
        inv_cov_mat_path += utils.data_fn_str(type='icovar', instr=instr, band=band, id='all', set=set, notes=notes)
        return inv_cov_mat_path

    def _read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        inv_cov_mat = enmap.read_map(fn)
        assert inv_cov_mat.ndim == 4, \
            'Inverse covariance matrices for a single dataset must have shape (npol, npol, ny, nx)'
        inv_cov_mat = inv_cov_mat[np.ix_(self._polidxs, self._polidxs)] * self._mult_fact
        inv_cov_mat = utils.atleast_nd(inv_cov_mat, 4)
        return {'inv_cov_mat': inv_cov_mat}

    def get_sim(self, split_num, sim_num):
        seed = (split_num, sim_num)
        seed += utils.hash_str(self._instr)
        seed += utils.hash_str(self._band)
        seed += utils.hash_str(self._id)
        seed += utils.hash_str(self._set)

        eta = utils.concurrent_standard_normal(
            size=(self._shape[1:]), seed=seed, dtype=self._dtype
            )
        return self.filter(eta, power=0.5)

    def filter(self, imap, power=-1):
        assert imap.shape == self._shape[1:], \
            f'Covariance matrix has shape {self._shape}, so imap must have shape {self._shape[1:]}'
        if power == -1:
            model = self.model
        else:
            try:
                # so we don't have to recompute if filtering by the same power later
                model = self._nm_dict[power]
            except KeyError:
                # because self.model already has -1 in exponent
                model = utils.eigpow(self.model, -power, axes=[-4, -3])
                self._nm_dict[power] = model
        return np.einsum('ab...,b...->a...', model, imap)

    @property
    def model(self):
        return self._nm_dict['inv_cov_mat']

    @model.setter
    def model(self, value):
        assert value.shape == self._shape, \
            f'Set shape is {value.shape}, expected {self._shape}'
        self._nm_dict['inv_cov_mat'] = value