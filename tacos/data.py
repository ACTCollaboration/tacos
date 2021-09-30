#!/usr/bin/env python3

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

    def __init__(self, instr, band, id=None, set=None, notes=None, pysm_notes=None, correlated_noise=False, pysm=False, 
                    healpix=False, cmb=False, cmb_kwargs=None, noise=False, noise_kwargs=None, 
                    beam_kwargs=None, bandpass_kwargs=None, **kwargs):
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
        noise : bool, None, int, or tuple-of-int
            Whether to add a noise realization to the map. False will pass; None, int, or
            tuple-of-int will be set as the seed of the realization. True will set cmb to
            None. Raises exception if pysm is False.
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
        if set is None:
            set = 'coadd'
        self.set = set
        self.notes = notes
        self.correlated_noise = correlated_noise
        self.pysm = pysm
        self.healpix = healpix

        if self.correlated_noise:
            raise NotImplementedError('Correlated noise not yet implemented')
        else:
            self.covmat_type = 'icovar'

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

        covmat_path = utils.data_dir_str('covmat', instr)
        covmat_path += utils.data_fn_str(type=self.covmat_type, instr=instr, band=band, id=id, set=set, notes=notes)
        
        # add optional mult_fact from noise_kwargs
        noise_kwargs = {} if noise_kwargs is None else noise_kwargs
        mult_fact = noise_kwargs.get('mult_fact', 1)
        self._covmat = mult_fact * enmap.read_map(covmat_path) # (npol, npol, ny, nx)

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

        if noise is not False:
            assert pysm, 'Can only add a noise realization to a simulated map'
            if noise is True:
                noise = None
            if self.covmat_type == 'icovar':
                self._map += utils.get_icovar_noise_sim(icovar=self.covmat, seed=noise)
            else:
                raise NotImplementedError('Correlated noise not yet implemented')

    def convolve_to_beam(self, bell):
        pass

    def convolve_with_beam(self, bell):
        pass

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