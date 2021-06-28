from ast import literal_eval

import numpy as np

from pixell import enmap

from tacos import utils, mixing_matrix as M

class Params:

    def __init__(self, components, shape, wcs=None, dtype=np.float32):
        
        ncomp = len(list(components))
        
        self.shape = shape
        utils.check_shape(self.shape)
        
        self._wcs = wcs # if this is None, return array as-is (ie, healpix), see amplitudes property
        self._dtype = dtype

        # initialize amplitudes
        # amplitudes are a single array, with outermost axis for components
        self._amplitudes = np.zeros((ncomp,) + self.shape, dtype=dtype)

        # for each component, store parameters that are either physically distinct or don't broadcast together
        self._params = {
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
                self.params[comp.name][active_param] = np.zeros(shape, dtype=dtype)
        assert len(self.params) == ncomp, 'At least one component has a repeated name, this is not allowed'

    @classmethod
    def load_from_config(cls, config_path, verbose=False):
        _, components, _, shape, wcs, kwargs = M._load_all_from_config(config_path, load_channels=False, verbose=True)
        return cls(components, shape, wcs, **kwargs)
       
    @property
    def amplitudes(self):
        if self._wcs is None:
            return self._amplitudes
        else:
            return enmap.ndmap(self._amplitudes, self._wcs)

    @amplitudes.setter
    def amplitudes(self, arr):
        self._amplitudes[:] = arr

    @property
    def params(self):
        return self._params