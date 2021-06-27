from ast import literal_eval

import numpy as np

from pixell import enmap

from tacos import utils, mixing_matrix as M

class Params:

    def __init__(self, components, shape, wcs, dtype=np.float32):
        self.ncomp = len(list(components))
        self.shape = shape
        self.check_shape()
        
        # initialize amplitudes
        # amplitudes are a single array, with outermost axis for components
        self._amplitudes = enmap.zeros((self.ncomp,) + self.shape, wcs, dtype=dtype)

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

        self.check_params()

    @classmethod
    def load_from_config(cls, config_path, verbose=False):
        _, components, shape, wcs, kwargs = M.load_mixing_matrix_init_from_config(config_path, load_channels=False, verbose=True)
        return cls(components, shape, wcs, **kwargs)

    def get_pol_indices(self, pol):
        return tuple('IQU'.index(p) for p in pol)

    def check_shape(self):
        assert len(self.shape) == 3 or len(self.shape) == 2 # car and healpix
        assert self.shape[-len(self.shape)] in (1,2,3), 'Polarization must have 1, 2, or 3 components'

    def check_params(self):
        assert len(self.params) == self.ncomp, 'At least one component has a repeated name, this is not allowed'
       
    @property
    def amplitudes(self):
        return self._amplitudes

    @property
    def params(self):
        return self._params