import numpy as np

from pixell import enmap

from tacos import utils

class Params:

    def __init__(self, shape, wcs, components, dtype=np.float32):
        self.ncomp = len(list(components))
        
        # initialize amplitudes
        self.check_shape(shape)
        shape = (self.ncomp,) + shape
        self._amplitudes = enmap.zeros(shape, wcs, dtype=dtype)

        # for each component, store parameters that are either physically distinct or don't broadcast together.
        # this must be implemented in each component, as a mappable under comp.params['active'].
        # for convenience, also stored the fixed parameters (comp.params['fixed'])
        self._params = {comp.name: comp.params for comp in components}
        self.check_params()

    @classmethod
    def load_from_config(cls, config_path):
        config = utils.config_from_yaml_file(config_path)

    def load_component_from_config():
        pass

    def check_shape(self, shape):
        assert len(shape) == 3
        assert shape[0] in (1,2,3), 'Only 1, 2, or 3 polarization components implemented'

    def check_params(self):
        assert len(self.params) == self.ncomp, 'At least one component has a repeated name, this is not allowed'
        for k, v in self.params.items():
            assert len(v) == 2, f'Component {k} params does not contain exactly "active" and "fixed" keys'
            assert 'active' in v, f'Component {k} params does not contain "active" key, contains {v} instead'
            assert 'fixed' in v, f'Component {k} params does not contain "fixed" key, contains {v} instead'

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, name, param, val):
        self._params[name]['active'][param] = val