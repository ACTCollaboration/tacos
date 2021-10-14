from pixell import enmap, wcsutils
import h5py
import numpy as np

from tacos import utils, config

import warnings
import os


chain_path = utils.data_dir_str('chain')


class Chain:

    def __init__(self, components, shape, wcs=None, dtype=None, fname=None, name=None, max_N=None):
        
        ncomp = len(components)
        self._components = components
        self._shape = shape
        utils.check_shape(self._shape)
        self._wcs = wcs # if this is None, return array as-is (ie, healpix), see amplitudes property
        self._dtype = dtype if dtype else 1000
        self._fname = self._get_fname(fname=fname, name=name)
        self._name = name # used for writing to disk if fname not provided
        self._max_N = max_N if max_N else 1000 # maximum array size. once reached, chain is dumped
        self._N = 0 # current sample counter

        # initialize chain info, this is just weights and -2logposts
        self._weights = np.full((self._max_N, 2), np.nan, dtype=self._dtype)

        # initialize amplitudes
        # amplitudes are a single array, with outermost axis for components
        self._amplitudes = np.full((self._max_N, ncomp, *self.shape), np.nan, dtype=self._dtype)
        if wcs is not None:
            self._amplitudes = enmap.ndmap(self._amplitudes, wcs=wcs)

        # for each component with active parameters, store parameters separately.
        self._params = self.get_empty_params_sample()
        for comp, param in self.paramsiter(yield_component=True):
            if param in comp.param_shapes:
                shape = comp.param_shapes[param]
            else:
                shape = self.shape
            self._params[comp.name][param] = np.full((self._max_N, *shape), np.nan, dtype=self._dtype)

    def _get_fname(self, fname=None, name=None):
        # allow the chain name to be the filename
        if fname is not None:
            return fname
        elif name is not None:
            fname = chain_path
            if name[-5:] != '.hdf5':
                name += '.hdf5'
            fname += name
            return fname
        else:
            raise ValueError('Chain has no name; must supply fullpath filename to dump to')

    def paramsiter(self, yield_component=False):
        for comp in self._components:
            for param in comp.active_params:
                if yield_component:
                    yield comp, param
                else:
                    yield comp.name, param

    def get_empty_params_sample(self):
        d = {}
        for comp, param in self.paramsiter():
            if comp not in d:
                d[comp] = {}
            d[comp][param] = None
        return d

    def add_samples(self, weights=None, amplitudes=None, params=None):
        assert not(weights is None and amplitudes is None and params is None), \
            'All of weights, amplitudes, and params passed as None'

        if self._N > 0:
            prev_weights, prev_amplitudes, prev_params = self.get_samples(sel=np.s_[-1 % self._N])
        
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

        # (1) check all samples are finite
        # (2) check all samples have the same length and that we can store the number of samples 
        # in this Chain.
        # (3) check if we can add samples given current counter.
        # (3a) If so, add them.
        # (3b) If not, or if after adding them we reach max counter, then dump.
        self._check_samples(weights, amplitudes, params)
        delta_N = self._get_sample_length(weights, amplitudes, params)
        assert delta_N <= self._max_N, 'Cannot add more samples than this Chain can ever hold'
        
        if self._N + delta_N <= self._max_N:      
            self._add_weights(weights, delta_N)
            self._add_amplitudes(amplitudes, delta_N)
            for comp, param in self.paramsiter():
                self._add_params(comp, param, params[comp][param], delta_N)
                        
            self._N += delta_N
            if self._N == self._max_N:
                warnings.warn(f'Sample counter reached max of {self._max_N}; writing to {self._fname} and reseting')
                self.write_samples(overwrite=False)
        else:
            warnings.warn(f'Sample counter would reach max of {self._max_N} if samples added; ' + \
                f'writing to {self._fname} first, reseting, and then adding samples')
            self.write_samples(overwrite=False)
            self.add_samples(weights=weights, amplitudes=amplitudes, params=params)

    def get_samples(self, sel=None):
        if sel is None:
            assert self._N > 0, 'Cannot get samples from empty chain'
            sel = np.s_[:self._N]
        return (
            self.get_weights(sel=sel), self.get_amplitudes(sel=sel), self.get_params(sel=sel)
        )

    def get_weights(self, sel=None):
        if sel is None:
            assert self._N > 0, 'Cannot get samples from empty chain'
            sel = np.s_[:self._N]
        return self._weights[sel]

    def get_amplitudes(self, sel=None):
        if sel is None:
            assert self._N > 0, 'Cannot get samples from empty chain'
            sel = np.s_[:self._N]
        return self._amplitudes[sel]

    def get_params(self, sel=None):
        if sel is None:
            assert self._N > 0, 'Cannot get samples from empty chain'
            sel = np.s_[:self._N]
        d = self.get_empty_params_sample()
        for comp, param in self.paramsiter():
            d[comp][param] = self._params[comp][param][sel]
        return d 

    def _check_samples(self, weights, amplitudes, params):
        assert np.isfinite(weights).sum() == weights.size, 'Weights contains a non-finite entry'
        assert np.isfinite(amplitudes).sum() == amplitudes.size, 'Amplitudes contains a non-finite entry'
        for comp, param in self.paramsiter():
            assert np.isfinite(params[comp][param]).sum() == params[comp][param].size, \
                f'{comp} param {param} contains a non-finite entry'

    def _get_sample_length(self, weights, amplitudes, params):
        max_length = len(weights) # this is 1 or N

        # check that len(amplitudes) is 1 or N
        if len(amplitudes) > max_length:
            assert max_length == 1, \
                'Weights and amplitudes have >1 sample and different number of samples'
            max_length = len(amplitudes)
        elif len(amplitudes) < max_length:
            assert len(amplitudes) == 1, \
                'Weights and amplitudes have >1 sample and different number of samples'

        # check that len(each param) is 1 or N
        # NOTE: if no active params, this loop does nothing
        for comp, param in self.paramsiter():
            if len(params[comp][param]) > max_length:
                assert max_length == 1, \
                    f'Comp {comp} param {param} have >1 sample and different number of samples ' + \
                    'than either weights, amplitudes, or a previous param'
                max_length = len(params[comp][param])
            elif len(params[comp][param]) < max_length:
                assert len(params[comp][param]) == 1, \
                    f'Comp {comp} param {param} have >1 sample and different number of samples ' + \
                    'than either weights, amplitudes, or a previous param'

        return max_length

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

    def write_samples(self, fname=None, name=None, overwrite=False):
        # allow the chain name to be the filename. fname (fullpath) takes
        # precedence over name
        if fname is None and name is None:
            fname = self._fname
        else:
            fname = self._get_fname(fname, name)

        # get all the samples; this means up to the current counter. beyond
        # the current counter is always garbage. check this!
        weights, amplitudes, params = self.get_samples(sel=np.s_[:self._N])
        self._check_samples(weights, amplitudes, params)

        if overwrite or not os.path.exists(fname):
            with h5py.File(fname, 'w') as hfile:
                # put all the samples in a 'samples' group, in case there is metadata to attach later
                # that is not specific to either the weights, amplitudes, or params
                hgroup = hfile.create_group('samples')

                # create new datasets with 1 chunk per sample and unlimited future samples
                hgroup.create_dataset(
                    'weights',
                    data=weights,
                    chunks=(1, *weights.shape[1:]),
                    maxshape=(None, *weights.shape[1:])
                    )
                hgroup.create_dataset(
                    'amplitudes',
                    data=amplitudes,
                    chunks=(1, *amplitudes.shape[1:]),
                    maxshape=(None, *amplitudes.shape[1:])
                    )
                for comp, param in self.paramsiter():
                    hgroup.create_dataset(
                        f'params/{comp}/{param}',
                        data=params[comp][param],
                        chunks=(1, *params[comp][param].shape[1:]),
                        maxshape=(None, *params[comp][param].shape[1:])
                        )
        else:
            with h5py.File(fname, 'r+') as hfile:
                # resize all the datasets by the number of samples we are adding
                weights_dset = hfile['samples/weights']
                weights_dset.resize(weights_dset.shape[0] + self._N, axis=0)

                amplitudes_dset = hfile['samples/amplitudes']
                amplitudes_dset.resize(amplitudes_dset.shape[0] + self._N, axis=0)

                params_dset = self.get_empty_params_sample()
                for comp, param in self.paramsiter():
                    params_dset[comp][param] = hfile[f'samples/params/{comp}/{param}']
                    params_dset[comp][param].resize(params_dset[comp][param].shape[0] + self._N, axis=0)

                # add samples to datasets
                weights_dset.write_direct(weights, dest_sel=np.s_[-self._N:])
                amplitudes_dset.write_direct(amplitudes, dest_sel=np.s_[-self._N:])
                for comp, param in self.paramsiter():
                    params_dset[comp][param].write_direct(params[comp][param], dest_sel=np.s_[-self._N:])

        self.__init__(
            self._components, self._shape, wcs=self._wcs, dtype=self._dtype,
            fname=self._fname, name=self._name, max_N=self._max_N
            )

    def read_samples(self, fname=None, name=None):
        # allow the chain name to be the filename. fname (fullpath) takes
        # precedence over name
        if fname is None and name is None:
            fname = self._fname
        else:
            fname = self._get_fname(fname, name)

        # currently this reads all samples
        with h5py.File(fname, 'r') as hfile:
            weights_dset = hfile['samples/weights']
            weights = np.empty(weights_dset.shape, weights_dset.dtype)
            weights_dset.read_direct(weights)

            amplitudes_dset = hfile['samples/amplitudes']
            amplitudes = np.empty(amplitudes_dset.shape, amplitudes_dset.dtype)
            amplitudes_dset.read_direct(amplitudes)

            params = self.get_empty_params_sample()
            for comp, param in self.paramsiter():
                params_dset = hfile[f'samples/params/{comp}/{param}']
                params[comp][param] = np.empty(params_dset.shape, params_dset.dtype)
                params_dset.read_direct(params[comp][param])
        
        self.add_samples(weights=weights, amplitudes=amplitudes, params=params)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self):
        return self._name

    @classmethod
    def load_from_config(cls, config_path, verbose=True):
        config_obj = config.Config(config_path, load_channels=False, verbose=verbose)
        components = config_obj.components
        shape = config_obj.shape
        wcs = config_obj.wcs
        dtype = config_obj.dtype
        name = config_obj.name
        max_N = config_obj.max_N
        return cls(components, shape, wcs=wcs, dtype=dtype, name=name, max_N=max_N)