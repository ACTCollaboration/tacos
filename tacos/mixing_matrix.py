import numpy as np
from scipy.interpolate import interp1d
import os 

from pixell import enmap
import h5py

from tacos import units, utils, models, broadcasting

models_config = utils.config_from_yaml_resource('configs/models.yaml')

class Component:

    def __init__(self, model, name=None, broadcasters=None, **kwargs):
        
        self._model = model
        self._name = name if name else model.__class__.__name__.lower()

        self._params = {
            'active': {},
            'fixed': {}
        }

        # this way, if a broadcaster is not set, we just broadcast it to itself
        self._broadcasters = {param: lambda x: x for param in model.params}

        # store fixed and active parameters in _params
        for param in model.params:
            val = kwargs.get(param)
            if val is not None:
                self.set_fixed_param(param, val)
            else:
                self.set_active_param(param, val)

        # store broadcasters, and evaluate them (once) for any fixed params
        if broadcasters is None:
            broadcasters = {}
        for param in broadcasters:
            if param in self.fixed_params:
                print(f'Broadcasting fixed param {param} and overwriting its value to broadcasted result')
                self.set_fixed_param(param, broadcasters[param](self.get_fixed_param(param)))
            elif param in self.active_params:
                self.broadcasters[param] = broadcasters[param]

    def __call__(self, nu, **kwargs):

        # first broadcast active params
        for param in self.active_params:
            kwargs[param] = self.broadcasters[param](kwargs[param])

        # then add in the already-broadcasted fixed params
        kwargs.update(self.fixed_params)

        # finally, evaluate
        return self._model(nu, **kwargs)

    def get_fixed_param(self, param):
        return self.fixed_params[param]

    def set_fixed_param(self, param, value):
        self.fixed_params[param] = value

    def get_active_param(self, param):
        return self.active_params[param]

    def set_active_param(self, param, value):
        self.active_params[param] = value

    @property
    def params(self):
        return self._model.params

    @property
    def fixed_params(self):
        return self._params['fixed']
    
    @property
    def active_params(self):
        return self._params['active']

    @property
    def broadcasters(self):
        return self._broadcasters

    @classmethod
    def load_from_config(cls, config_path, name):

        # load the block under 'components' by name.
        # first look in the default configs, then assume 'config' is a full path
        try:
            comp_block = utils.config_from_yaml_resource(config_path)['components'][name]
        except FileNotFoundError:
            comp_block = utils.config_from_yaml_file(config_path)['components'][name]

        # first get the model of the component
        model_name = comp_block['model']
        nu0 = comp_block.get('nu0') # if not provided, use the default in models.py
        model_kwargs = {'nu0': float(nu0)} if nu0 else {}
        model = getattr(models, model_name)(**model_kwargs)      

        # if there are params, then:
        # for each parameter listed, determine if active or fixed and if fixed, then:
        # how to load the fixed value
        kwargs = {}
        broadcasters = {}
        if 'params' in comp_block:
            for param, info in comp_block['params'].items():
                assert param in model.params, f'Param {param} not in {model_name} params'
                kwargs = cls.load_fixed_param(kwargs, param, info, model_name=model_name)
                broadcasters = cls.load_broadcaster(broadcasters, param, info, model_name=model_name)

        # get component
        return cls(model, name, broadcasters=broadcasters, **kwargs)

    @staticmethod
    def load_fixed_param(kwargs, param, info, model_name=None):
        
        # fixed parameters specified by key 'value'
        if 'value' in info:
            value = info['value']
            assert 'shape' not in info, 'A fixed value cannot have a config-set shape'

            # if scalar, fix parameter to that value everywhere 
            if isinstance(value, (int, float)):
                print(f'Fixing {model_name} param {param} to {float(value)}')
                value = float(value)

            # if string, first see if it is in the config templates
            # if not, load it directly
            elif isinstance(value, str):
                if info in models_config['templates']:
                    print(f'Fixing {model_name} param {param} to {value} template')
                    value = enmap.read_map(models_config['templates'][value])
                else:
                    print(f'Fixing {model_name} param {param} to data at {value}')
                    value = enmap.read_map(value)
            
            # add it to the mapping we are building
            kwargs[param] = value
        return kwargs

    @staticmethod
    def load_broadcaster(broadcasters, param, info, model_name=None):
        
        # broadcasters specified by key 'broadcasters'
        if 'broadcasters' in info:
            
            # for each function in broadcasters, add it to the function call stack
            # with any kwargs as necessary
            func_list = []
            kwarg_list = []
            for func_name, func_kwargs in info['broadcasters'].items():
                print(f'Appending {func_name} to broadcasting call stack for {model_name} param {param}')
                broadcaster = getattr(broadcasting, func_name)
                func_list.append(broadcaster) 
                if func_kwargs == 'None':
                    kwarg_list.append({})
                else:
                    kwarg_list.append(func_kwargs)

            # build a single function call stack
            def stacked_broadcaster(x):
                for i in range(len(func_list)):
                    x = func_list[i](x, **kwarg_list[i])
                return x

            # add it to the mapping we are building
            broadcasters[param] = stacked_broadcaster
        return broadcasters

class Element:

    def __init__(self, channel, component, active_params=None, broadcaster_dict=None, interpolation_kwargs=None):
        # broadcaster_dict needs to specify function and kwargs for each active param
        # in the component.
        pass 


def get_mixing_matrix(channels, components, dtype=np.float32):
    '''
    Return mixing matrix for given frequency bands and signal
    components.

    Parameters
    ----------
    bandpasses : (nj) array-like of BandPass objects
        Bandpass for each frequency band.
    betas : (ncomp) array or (ncomp, ny, nx) enmap.
        Spectral indices per components.
    dtype : type
        Dtype for output.
    
    Returns
    -------
    mixing_mat : (nj, ncomp, ...) array or ndmap
        Mixing matrix.
    '''

    nchan = len(channels)
    ncomp = len(components)

    if hasattr(channels[0].map, 'wcs'):
        is_enmap = True
        wcs = channels[0].map.wcs
    else:
        is_enmap = False

    m_shape = (nchan, ncomp) + channels[0].map.shape
    m = np.zeros(m_shape, dtype=dtype)

    for chanidx, chan in enumerate(channels):
        u_conv = chan.bandpass.rj_to_cmb
        for compidx, comp in enumerate(components):
            res = u_conv * chan.bandpass.integrate_signal(comp)
            m[chanidx, compidx] = res

    if is_enmap:
        m = enmap.ndmap(m, wcs)

    return m