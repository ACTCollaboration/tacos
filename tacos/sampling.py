from ast import literal_eval

import numpy as np
from pixell import enmap
from astropy.io import fits

from tacos import utils, mixing_matrix as M

class Chain:

    @classmethod
    def load_from_config(cls, config_path, verbose=False):
        _, components, _, shape, wcs, kwargs = M._load_all_from_config(config_path, load_channels=False, verbose=verbose)
        return cls(components, shape, wcs, **kwargs)

    def __init__(self, components, shape, wcs=None, dtype=np.float32, fname=None):
        
        ncomp = len(list(components))
        
        self.shape = shape
        utils.check_shape(self.shape)
        
        self._wcs = wcs # if this is None, return array as-is (ie, healpix), see amplitudes property
        self._dtype = dtype
        self._fname = fname

        # initialize chain info, this is just weights and -2logposts
        self._weights = []
        self._weights_shape = (2,)

        # initialize amplitudes
        # amplitudes are a single array, with outermost axis for components
        self._amplitudes = []
        self._amplitudes_shape = (ncomp,) + self.shape

        # for each component with active parameters, store parameters separately.
        # store shapes in a similar structure
        self._params = {}
        self._params_shape = {}
        for comp in components:
            assert comp.name not in self._params and comp.name not in self._params_shape, \
                f'Component {comp} has repeated name'
            self._params[comp.name] = {}
            self._params_shape[comp.name] = {}
            for active_param in comp.active_params:
                if active_param in comp.shapes:
                    shape = comp.shapes[active_param]
                else:
                    shape = self.shape
                self._params[comp.name][active_param] = []
                self._params_shape[comp.name][active_param] = shape

        assert len(self._params) == ncomp, \
            'At least one component has a repeated name, this is not allowed'
        self._check_lengths()

    def _check_lengths(self, delta_length=1):
        # we want to make sure no chain is more than delta_length longer than any other.
        # you probably want delta_length=1 when updating parameters, and delta_length=0
        # when writing to disk
        lengths = np.array(self._get_lengths())
        diff = lengths.max() - lengths.min()
        assert diff <= delta_length, \
            f'The max. diff. in num. of iterations between chain objects is {diff}; expected {delta_length}'

    def _get_lengths(self):
        lengths = (len(self.weights), len(self.amplitudes)) + self._get_params_lengths()
        return lengths

    def _get_params_lengths(self):
        lengths = ()
        for active_params in self._params.values():
            lengths += tuple(len(param) for param in active_params.values())
        return lengths

    def append_weights(self, weights):
        weights = np.asarray(weights, dtype=self._dtype)
        assert weights.shape == self._weights_shape, \
            f'Attempted to append weights with shape {weights.shape}; expected {self._weights_shape}'
        self._weights.append(weights)
        self._check_lengths()

    def get_weights(self, iteration=-1):
        return self._weights[iteration]

    def append_amplitudes(self, amplitudes):
        amplitudes = np.asarray(amplitudes, dtype=self._dtype)
        assert amplitudes.shape == self._amplitudes_shape, \
            f'Attempted to append amplitudes with shape {amplitudes.shape}; expected {self._amplitudes_shape}'
        self._amplitudes.append(amplitudes)
        self._check_lengths()

    def get_amplitudes(self, iteration=-1):
        return self._amplitudes[iteration]

    def append_params(self, comp_name, param_name, params):
        params = np.asarray(params, dtype=self._dtype)
        assert params.shape == self._params_shape[comp_name][param_name], \
            f'Attempted to append {comp_name} param {param_name} with shape {params.shape}; expected {self._params_shape[comp_name][param_name]}'
        self._params[comp_name][param_name].append(params)
        self._check_lengths()

    def get_params(self, iteration=-1):
        return self._params[iteration]

    def get_all_weights(self):
        return np.asarray(self._weights, dtype=self._dtype)
       
    def get_all_amplitudes(self):
        if self._wcs is None:
            return np.asarray(self._amplitudes, dtype=self._dtype)
        else:
            return enmap.enmap(self._amplitudes, self._wcs, dtype=self._dtype, copy=False)

    def get_all_params(self):
        out = {}
        for comp_name in self._params:
            out[comp_name] = {}
            for active_param in out[comp_name]:
                out[comp_name][active_param] = np.asarray(self._params[comp_name][active_param], dtype=self._dtype)
        return self._params

    def dump(self, fname=None):
        if fname is None:
            fname = self._fname

