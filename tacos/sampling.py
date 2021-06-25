from ast import literal_eval

import numpy as np

from pixell import enmap

from tacos import utils, mixing_matrix as M

class Params:

    def __init__(self, components, pol, shape, wcs, dtype=np.float32):
        self.ncomp = len(list(components))
        self.npol = len(pol)
        self.pol_idxs = self.get_pol_indices(pol)
        
        # initialize amplitudes
        self.shape = (self.ncomp, self.npol) + shape
        self.check_shape()
        self._amplitudes = enmap.zeros(self.shape, wcs, dtype=dtype)

        # for each component, store parameters that are either physically distinct or don't broadcast together
        # additionally, initialize them
        self._params = {
            comp.name: {
                active_param: None for active_param in comp.active_params
                } 
            for comp in components
            }
        for comp in components:
            for active_param in comp.active_params:
                if active_param in comp.shapes:
                    shape = comp.shapes[active_param]
                else:
                    shape = self.shape
                self.params[comp.name][active_param] = np.zeros(shape, dtype=dtype)
        self.check_params()

    @classmethod
    def load_from_config(cls, config_path):
        try:
            config = utils.config_from_yaml_resource(config_path)
        except FileNotFoundError:
            config = utils.config_from_yaml_file(config_path)
            
        params_block = config['parameters']
        components_block = config['components']

        # get pol, shape, wcs, dtype
        pol = params_block['pol']
        shape, wcs = enmap.read_map_geometry(params_block['geometry'])
        shape = shape[-2:]
        kwargs = {'dtype': params_block.get('dtype')} if params_block.get('dtype') else {}

        # get the components
        # we can say verbose is False because we are just piggy-backing off the component load_from_config
        # method to populate component name and active params; we don't care about other info
        components = [M.Component.load_from_config(config_path, comp, verbose=False) for comp in components_block]

        return cls(components, pol, shape, wcs, **kwargs)


    def get_pol_indices(self, pol):
        return tuple('IQU'.index(p) for p in pol)

    def check_shape(self):
        assert len(self.shape) == 4
        assert self.shape[-3] in (1,2,3), 'Polarization must have 1, 2, or 3 components'

    def check_params(self):
        assert len(self.params) == self.ncomp, 'At least one component has a repeated name, this is not allowed'
       
    @property
    def amplitudes(self):
        return self._amplitudes

    @property
    def params(self):
        return self._params