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
        try:
            config = utils.config_from_yaml_resource(config_path)
        except FileNotFoundError:
            config = utils.config_from_yaml_file(config_path)
            
        params_block = config['parameters']
        components_block = config['components']

        # get pol, shape, wcs, dtype
        pol = params_block['pol']
        shape, wcs = enmap.read_map_geometry(params_block['geometry'])
        shape = (len(pol),) + shape[-2:]
        kwargs = {'dtype': params_block.get('dtype')} if params_block.get('dtype') else {}

        # get the components
        # we can say verbose is False because we are just piggy-backing off the component load_from_config
        # method to populate component name and active params; we don't care about other info
        components = [M.Component.load_from_config(config_path, comp, verbose=verbose) for comp in components_block]

        return cls(components, shape, wcs, **kwargs)

    def get_pol_indices(self, pol):
        return tuple('IQU'.index(p) for p in pol)

    def check_shape(self):
        assert len(self.shape) == 3
        assert self.shape[-3] in (1,2,3), 'Polarization must have 1, 2, or 3 components'

    def check_params(self):
        assert len(self.params) == self.ncomp, 'At least one component has a repeated name, this is not allowed'
       
    @property
    def amplitudes(self):
        return self._amplitudes

    @property
    def params(self):
        return self._params