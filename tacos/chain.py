from pixell import enmap, wcsutils
from astropy.io import fits
import numpy as np

from tacos import utils, mixing_matrix as M

import warnings

config = utils.config_from_yaml_resource('configs/sampling.yaml')

class Chain:

    @classmethod
    def load_from_config(cls, config_path, verbose=False):
        name, _, components, _, shape, wcs, kwargs = M._load_all_from_config(config_path, load_channels=False, verbose=verbose)
        return cls(components, shape, wcs, name=name, **kwargs)

    def __init__(self, components, shape, wcs=None, dtype=np.float32, name=None,
                    max_N=1000, fname=None):
        
        ncomp = len(components)
        self._components = components
        self._shape = shape
        utils.check_shape(self._shape)
        self._wcs = wcs # if this is None, return array as-is (ie, healpix), see amplitudes property
        self._dtype = dtype
        self._name = name # used for writing to disk if fname not provided
        self._max_N = max_N # maximum array size. once reached, chain is dumped
        self._N = 0 # current sample counter
        self._fname = self._get_fname(fname, name) # used for autodumping

        # initialize chain info, this is just weights and -2logposts
        self._weights = np.empty((max_N, 2), dtype=dtype)

        # initialize amplitudes
        # amplitudes are a single array, with outermost axis for components
        if wcs is None:
            self._amplitudes = np.empty((max_N, ncomp, *self.shape), dtype=dtype)
        else:
            self._amplitudes = enmap.empty((max_N, ncomp, *self.shape), wcs=wcs, dtype=dtype)

        # for each component with active parameters, store parameters separately.
        self._params = self.get_empty_params_sample()
        for comp, param in self.paramsiter():
            if param in comp.shapes:
                shape = comp.shapes[param]
            else:
                shape = self.shape
            self._params[comp][param] = np.empty((max_N, *shape), dtype=dtype)

    def _get_fname(self, fname, name):
        # allow the chain name to be the filename
        if fname is not None:
            return fname
        elif name is not None:
            fname = config['output']['chain_path'] + name + '.fits'
            return fname
        else:
            raise ValueError('Chain has no name; must supply explicit filename to dump to')

    def paramsiter(self):
        for comp in self._components:
            for param in comp.active_params:
                yield comp.name, param

    def get_empty_params_sample(self):
        d = {}
        for comp in self._components:
            d[comp.name] = {}
        for comp, param in self.paramsiter():
            d[comp][param] = None
        assert len(d) == len(self._components), \
            'At least one component has a repeated name, this is not allowed'
        return d

    def add_samples(self, weights=None, amplitudes=None, params=None):
        if self._N > 0:
            prev_weights, prev_amplitudes, prev_params = self.get_samples()
        
        # copy forward the previous sample. this will fail if counter at 0 since
        # names would not exist
        if weights is None:
            weights = prev_weights
        if amplitudes is None:
            amplitudes = prev_amplitudes
        if params is None:
            params = prev_params

        # make at least the dimension of chain shapes, even if singleton in sample number
        weights = utils.atleast_nd(weights, self._weights.ndim)
        amplitudes = utils.atleast_nd(amplitudes, self._amplitudes.ndim)
        for comp, param in self.paramsiter():
            params[comp][param] = utils.atleast_nd(
                params[comp][param], self._params[comp][param].ndim
                )

        # check all samples have the same length and that we can store the number of samples 
        # in this Chain. If so, check if we can add samples given current counter. If so, add
        # them. If not, or if after adding them we reach max counter, then dump.
        delta_N = self._get_sample_length(weights, amplitudes, params)
        assert delta_N <= self._max_N, 'Cannot add more samples than this Chain can ever hold'
        
        if self._N + delta_N <= self._max_N:      

            # add sample and increment counter
            self._add_weights(weights, delta_N)
            self._add_amplitudes(amplitudes, delta_N)

            # iterate over internal components, active_params to enforce that 
            # kwarg params has supplied all of them
            for comp, param in self.paramsiter():
                self._add_params(comp, param, params[comp][param], delta_N)
            
            # if counter reaches max, dump
            self._N += delta_N

            # TODO: fix overwrite = Trues!!!!
            if self._N == self._max_N:
                warnings.warn(f'Sample counter reached max of {self._max_N}; writing to {self._fname} and reseting')
                self.write_samples(overwrite=True, reset=True)
        else:
            warnings.warn(f'Sample counter would reach max of {self._max_N} if samples added; ' + \
                f'writing to {self._fname}, reseting, and then adding samples')
            self.write_samples(overwrite=True, reset=True)
            self.add_sample(weights=weights, amplitudes=amplitudes, params=params)

    def get_samples(self, sel=None):
        if sel is None:
            assert self._N > 0
            sel = np.s_[-1 % self._N] # get most recent *written* element
        return (
            self._weights[sel], self._amplitudes[sel], self._get_params(sel)
        )

    def _get_params(self, sel):
        d = self.get_empty_params_sample()
        for comp, param in self.paramsiter():
            d[comp][param] = self._params[comp][param][sel]
        return d 

    def _get_sample_length(self, weights, amplitudes, params):
        equal_lengths = True
        equal_lengths &= len(weights) == len(amplitudes)

        # NOTE: if no active params, this loop does nothing
        for comp, param in self.paramsiter():
            equal_lengths &= len(weights) == len(params[comp][param])

        assert equal_lengths, 'Not all of weights, amplitudes, params have an equal number of samples'
        return len(weights)

    def _add_weights(self, weights, delta_N):
        weights = np.asarray(weights, dtype=self._dtype)
        assert weights.shape[1:] == self._weights.shape[1:], \
        f'Attempted to add weights with shape {weights.shape[1:]}; expected {self._weights.shape[1:]}'
        self._weights[self._N:self._N + delta_N] = weights

    def _add_amplitudes(self, amplitudes, delta_N):
        if self._wcs is not None:
            if hasattr(amplitudes, 'wcs'):
                assert wcsutils.is_compatible(self._wcs, amplitudes.wcs), \
                    'Attempted to add amplitudes with incompatible wcs to wcs of Chain'
            amplitudes = enmap.enmap(amplitudes, self._wcs, dtype=self._dtype, copy=False)
        else:
            amplitudes = np.asarray(amplitudes, dtype=self._dtype)
        assert amplitudes.shape[1:] == self._amplitudes.shape[1:], \
            f'Attempted to add amplitudes with shape {amplitudes.shape[1:]}; expected {self._amplitudes.shape[1:]}'
        self._amplitudes[self._N:self._N + delta_N] = amplitudes
    
    def _add_params(self, comp_name, param_name, params, delta_N):
        params = np.asarray(params, dtype=self._dtype)
        assert params.shape[1:] == self._params[comp_name][param_name].shape[1:], \
            f'Attempted to append {comp_name} param {param_name} with shape {params.shape[1:]}; ' + \
                f'expected {self._params[comp_name][param_name].shape[1:]}'
        self._params[comp_name][param_name][self._N:self._N + delta_N] = params

    def write_samples(self, fname=None, overwrite=True, reset=False):
        # allow the chain name to be the filename
        if fname is None:
            fname = self._fname

        # get all the samples
        weights, amplitudes, params = self.get_samples(sel=np.s_[:self._N])

        # first make the primary hdu. this will hold information informing us what the latter hdus are,
        # which is always amplitudes, followed be each active param.
        # the primary hdu itself holds the weights
        primary_hdr = fits.Header()
        primary_hdr['HDU1'] = 'AMPLITUDES'

        hidx = 2
        for comp, param in self.paramsiter():
            primary_hdr[f'HDU{hidx}'] = f'{comp}_{param}'.upper()
            hidx += 1

        primary_hdu = fits.PrimaryHDU(data=weights, header=primary_hdr)

        # next make the amplitude and params hdus. there is always an amplitude hdu (hdu 1)
        # but not always params (only if active params)
        if self._wcs is not None:
            amplitudes_hdr = self._wcs.to_header(relax=True) # if there is wcs information, retain it
        else:
            amplitudes_hdr = fits.Header()
        amplitudes_hdu = fits.ImageHDU(data=amplitudes, header=amplitudes_hdr)
        hdul = fits.HDUList([primary_hdu, amplitudes_hdu])

        for comp, param in self.paramsiter():
            hdul.append(fits.ImageHDU(data=params[comp][param]))
        
        # finally, write to disk and "reset" the chain to be empty
        # TODO: implement appending to the file, instead of overwriting it
        if overwrite:
            warnings.warn(f'Overwriting samples at {fname}')
            hdul.writeto(fname, overwrite=overwrite)
        else:
            raise NotImplementedError('Appending to disk not yet implemented')
        if reset:
            self.__init__(self._components, self._shape, wcs=self._wcs, dtype=self._dtype, name=self._name,
                        max_N=self._max_N, fname=self._fname)

    def read_samples(self, fname=None):

        # allow the chain name to be the filename
        if fname is None:
            fname = self._fname

        with fits.open(fname) as hdul:

            # get the primary hdu header, which refers to the subsequent hdus
            header = hdul[0].header

            # get the weights from the primary hdu
            weights = hdul[0].data

            # get the amplitudes from the 1st image hdu 
            assert header['HDU1'] == 'AMPLITUDES', 'HDU1 must hold AMPLITUDES'
            amplitudes = hdul[1].data

            # assign subsequent hdus to active params
            params = self.get_empty_params_sample()
            for hidx, (comp, param) in enumerate(self.paramsiter()):
                # we start from hidx = 2
                hidx += 2
            
                # check that the comp, params order on disk matches that of this Chain object
                comp_on_disk, param_on_disk = header[f'HDU{hidx}'].split('_')
                assert comp.upper() == comp_on_disk, \
                    'Chain loaded from config has different comp order than chain on disk'
                assert param.upper() == param_on_disk, \
                    'Chain loaded from config has different param order than chain on disk'

                # get the param from this hdu
                params[comp][param] = hdul[hidx].data
        
        self.add_samples(weights=weights, amplitudes=amplitudes, params=params)

    @property
    def shape(self):
        return self._shape

    @property
    def name(self):
        return self._name