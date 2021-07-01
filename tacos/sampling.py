from ast import literal_eval

import numpy as np

from pixell import enmap

from tacos import utils, mixing_matrix as M

class Chain:

    def __init__(self, components, shape, wcs=None, dtype=np.float32):
        
        ncomp = len(list(components))
        
        self.shape = shape
        utils.check_shape(self.shape)
        
        self._wcs = wcs # if this is None, return array as-is (ie, healpix), see amplitudes property
        self._dtype = dtype

        # initialize chain info, this is just weights and -2logposts
        self._weights = np.zeros((1, 2), dtype=dtype)

        # initialize amplitudes
        # amplitudes are a single array, with outermost axis for components
        self._amplitudes = np.zeros((1, ncomp) + self.shape, dtype=dtype)

        # for each component, store parameters that are either physically distinct or don't broadcast together
        self._params = {}
        for comp in components:
            if comp.active_paras
            comp.name: {
                active_param: None for active_param in comp.active_params
                } 
            for comp in components
            }

        # additionally, initialize them with zeros
        for comp in components:
            for active_param in comp.active_params:
                if active_param in comp.shapes:
                    shape = comp.shapes[active_param]
                else:
                    shape = self.shape
                
                # params are multiple arrays, one for each parameter, unlike amplitudes
                self.params[comp.name][active_param] = np.zeros((1,) + shape, dtype=dtype)
        assert len(self.params) == ncomp, 'At least one component has a repeated name, this is not allowed'

    def check_lengths(self, delta_length=1):
        # we want to make sure no chain is more than delta_length longer than any other.
        # you probably want delta_length=1 when updating parameters, and delta_length=0
        # when writing to disk
        lengths = np.array(self.get_lengths())
        print(lengths)
        return lengths.max() - lengths.min()

    def get_lengths(self):
        lengths = (len(self.weights), len(self.amplitudes)) + self._get_params_lengths()
        return lengths

    def _get_params_lengths(self):
        lengths = ()
        for active_params in self._params.values():
            lengths += tuple(len(param) for param in active_params)
        return lengths
    
    @classmethod
    def load_from_config(cls, config_path, verbose=False):
        _, components, _, shape, wcs, kwargs = M._load_all_from_config(config_path, load_channels=False, verbose=verbose)
        return cls(components, shape, wcs, **kwargs)

    @property
    def weights(self):
        return self._weights
       
    @property
    def amplitudes(self):
        if self._wcs is None:
            return self._amplitudes
        else:
            return enmap.ndmap(self._amplitudes, self._wcs)

    @property
    def params(self):
        return self._params