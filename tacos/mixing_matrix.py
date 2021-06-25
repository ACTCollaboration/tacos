import numpy as np
from scipy import interpolate as interp
import os 
from ast import literal_eval
import time

from pixell import enmap
import h5py

from tacos import units, utils, models, broadcasting

config = utils.config_from_yaml_resource('configs/mixing_matrix.yaml')

method_order_key = {
    'linear': 1,
    'quadratic': 2,
    'cubic': 3
} 

class Component:

    def __init__(self, model, name=None, comp_broadcaster=None, param_broadcasters=None, shapes=None, verbose=True, **kwargs):
        
        self._model = model
        self._name = name if name else model.__class__.__name__.lower()

        self._params = {
            'active': [],
            'fixed': {}
        }

        # this way, if a broadcaster is not set, we just broadcast it to itself
        self._broadcaster = lambda x: x
        self._broadcasters = {param: lambda x: x for param in model.params}

        self._shapes = {}

        # store fixed and active parameters in _params
        for param in model.params:
            val = kwargs.get(param)
            if val is not None:
                self.fixed_params[param] = val
            else:
                self.active_params.append(param)

        # store final broadcaster for component after evaluation of params
        if comp_broadcaster is not None:
            self.broadcaster = comp_broadcaster

        # store per-parameter broadcasters, and evaluate them (once) for any fixed params
        if param_broadcasters is None:
            param_broadcasters = {}
        for param in param_broadcasters:
            if param in self.fixed_params:
                print(f'Broadcasting fixed param {param} and overwriting its value to broadcasted result')
                self.fixed_params[param] = param_broadcasters[param](self.fixed_params[param])
            elif param in self.active_params:
                self.broadcasters[param] = param_broadcasters[param]

        # store shape for passing onto Params class, but it doesn't do anything here
        if shapes is None:
            shapes = {}
        for param, shape in shapes.items():
            if param in self.active_params:
                print(f'Storing active param {param} shape {shape}')
                self.shapes[param] = shapes[param]

    # This function oddly has no use when things are interpolated
    # I think it will come in handy when evaluating a proposal that has gone "out of bounds" TODO: implement that
    def __call__(self, nu, **kwargs):
        
        # first broadcast active params
        for param in self.active_params:
            kwargs[param] = self.broadcasters[param](kwargs[param])

        # then add in the already-broadcasted fixed params
        for param in self.fixed_params:
            assert param not in kwargs, f'Param {param} is fixed but was passed as a kwarg'
        kwargs.update(self.fixed_params)

        # finally, evaluate
        return self.model(nu, **kwargs)

    @classmethod
    def load_from_config(cls, config_path, name, verbose=True):

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
        model = getattr(models, model_name)(verbose=verbose, **model_kwargs)

        # get the component broadcaster, if any
        comp_broadcaster =  cls.load_broadcasters(name, comp_block, name, model_name)

        # if there are params, then:
        # for each parameter listed, determine if active or fixed and if fixed, then:
        # how to load the fixed value
        kwargs = {}
        param_broadcasters = {}
        shapes = {}
        if 'params' in comp_block:
            for param, info in comp_block['params'].items():
                assert param in model.params, f'Param {param} not in {model_name} params'
                kwargs = cls.load_fixed_param(kwargs, param, info, name, model_name)
                param_broadcasters = cls.load_broadcasters(param, info, name, model_name, broadcasters=param_broadcasters)
                shapes = cls.load_shape(shapes, param, info)

        # get component
        return cls(model, name, comp_broadcaster=comp_broadcaster,
                    param_broadcasters=param_broadcasters, shapes=shapes, verbose=verbose, **kwargs)

    # just helps modularize load_from_config(...)
    @classmethod
    def load_fixed_param(cls, kwargs, param, info, comp_name, model_name):

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
                if info in config['templates']:
                    print(f'Fixing component {comp_name} (model {model_name}) param {param} to {value} template')
                    value = enmap.read_map(config['templates'][value])
                else:
                    print(f'Fixing component {comp_name} (model {model_name}) param {param} to data at {value}')
                    value = enmap.read_map(value)
            
            # add it to the mapping we are building
            kwargs[param] = value
        return kwargs
    
    # just helps modularize load_from_config(...)
    @classmethod
    def load_broadcasters(cls, key, info, comp_name, model_name, broadcasters=None):

        # broadcasters specified by key 'broadcasters'
        if 'broadcasters' in info:        
            
            # for each function in broadcasters, add it to the function call stack
            # with any kwargs as necessary
            func_list = []
            kwarg_list = []
            for func_name, func_kwargs in info['broadcasters'].items():
                if broadcasters is not None: # for parameters
                    print(f'Appending {func_name} to broadcasting call stack for component {comp_name} (model {model_name}) param {key}')
                else: # for component
                    print(f'Appending {func_name} to broadcasting call stack for component {key}  (model {model_name})')
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
            if broadcasters is not None:
                broadcasters[key] = stacked_broadcaster
            else:
                broadcasters = stacked_broadcaster
        return broadcasters
    
    # just helps modularize load_from_config(...)
    @classmethod
    def load_shape(cls, shapes, param, info):
        # shapes specified by key 'shapes'
        if 'shape' in info:
            shape = info['shape']
            assert 'value' not in info, 'A shaped-config component cannot have a fixed value'

            shapes[param] = literal_eval(shape) # this maps a stringified tuple to the actual tuple
        return shapes

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._name

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
    def broadcaster(self):
        return self._broadcaster 

    @broadcaster.setter
    def broadcaster(self, value):
        self._broadcaster = value

    @property
    def broadcasters(self):
        return self._broadcasters

    @property
    def shapes(self):
        return self._shapes

class Element:

    def __init__(self, channel, component):
        
        self.channel = channel
        self.component = component
        self.method = config['interpolation']['method']
        self.u_conv = channel.bandpass.rj_to_cmb
        order = method_order_key[self.method]
        print(self.method)
        print(order)

        # get span of each param(s) in the component
        model_name = component.model.__class__.__name__

        self.spans = {}
        for param in component.params:
            comp_block = config['interpolation'][model_name][param]
            low = comp_block['low']
            high = comp_block['high']
            N = comp_block['N']
            self.spans[param] = np.linspace(low, high, N) # follows params order
        print(np.array(list(self.spans.values())).shape)

        # build interpolator
        nu = channel.bandpass.nu
        if len(self.spans) == 0:
            signal = component.model(nu)
            y = channel.bandpass.integrate_signal(signal) # this is one number!
            interpolator = lambda x: y

        elif len(self.spans) == 1:
            signal = component.model(nu, **self.spans)
            y = channel.bandpass.integrate_signal(signal)
            def interpolator(xx):
                f = interp.interp1d(*self.spans.values(), y, kind=self.method)
                return f(xx)

        elif len(self.spans) == 2:
            meshed_spans = np.meshgrid(*self.spans.values(), indexing='ij', sparse=True)
            meshed_spans = {k: v for k, v in zip(self.spans.keys(), meshed_spans)}
            signal = component.model(nu, **meshed_spans)
            y = channel.bandpass.integrate_signal(signal)
            def interpolator(xx, yy):
                f = interp.RectBivariateSpline(*self.spans.values(), y, kx=order, ky=order)
                return f(xx, yy, grid=False)
        else:
            raise NotImplementedError('Only up to 2-parameter models implemented so far')

        self.interpolator = interpolator

    def __call__(self, **kwargs):

        # we need to build a spans dictionary in the proper order
        spans = []
        for param in self.component.params:
        
            # broadcast active params, or grab already-broadcasted fixed params
            if param in self.component.active_params:
                spans.append(self.component.broadcasters[param](kwargs[param]))
            else:
                assert param not in kwargs, f'Param {param} is fixed but was passed as a kwarg'
                spans.append(self.component.fixed_params[param])

        # interpolate, broadcast with component broadcaster
        res = self.interpolator(*spans)
        res = self.u_conv * self.component.broadcaster(res)
        return res




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