import numpy as np
from scipy import interpolate as interp
from ast import literal_eval

from pixell import enmap

from tacos import data, utils, models, broadcasting

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
                if verbose: 
                    print(f'Broadcasting fixed param {param} and overwriting its value to broadcasted result')
                self.fixed_params[param] = param_broadcasters[param](self.fixed_params[param])
            elif param in self.active_params:
                self.broadcasters[param] = param_broadcasters[param]

        # store shape for passing onto Params class, but it doesn't do anything here
        if shapes is None:
            shapes = {}
        for param, shape in shapes.items():
            if param in self.active_params:
                if verbose:
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
        comp_broadcaster =  cls.load_broadcasters(name, comp_block, name, model_name, verbose=verbose)

        # if there are params, then:
        # for each parameter listed, determine if active or fixed and if fixed, then:
        # how to load the fixed value
        kwargs = {}
        param_broadcasters = {}
        shapes = {}
        if 'params' in comp_block:
            for param, info in comp_block['params'].items():
                assert param in model.params, f'Param {param} not in {model_name} params'
                kwargs = cls.load_fixed_param(kwargs, param, info, name, model_name, verbose=verbose)
                param_broadcasters = cls.load_broadcasters(param, info, name, model_name, broadcasters=param_broadcasters, verbose=verbose)
                shapes = cls.load_shape(shapes, param, info, name, model_name, verbose=verbose)

        # get component
        return cls(model, name, comp_broadcaster=comp_broadcaster,
                    param_broadcasters=param_broadcasters, shapes=shapes, verbose=verbose, **kwargs)

    # just helps modularize load_from_config(...)
    @classmethod
    def load_fixed_param(cls, kwargs, param, info, comp_name, model_name, verbose=True):

        # fixed parameters specified by key 'value'
        if 'value' in info:
            value = info['value']
            assert 'shape' not in info, 'A fixed value cannot have a config-set shape'

            # if scalar, fix parameter to that value everywhere 
            if isinstance(value, (int, float)):
                if verbose:
                    print(f'Fixing {model_name} param {param} to {float(value)}')
                value = float(value)

            # if string, first see if it is in the config templates
            # if not, load it directly
            elif isinstance(value, str):
                if info in config['templates']:
                    if verbose:
                        print(f'Fixing component {comp_name} (model {model_name}) param {param} to {value} template')
                    value = enmap.read_map(config['templates'][value])
                else:
                    if verbose:
                        print(f'Fixing component {comp_name} (model {model_name}) param {param} to data at {value}')
                    value = enmap.read_map(value)
            
            # add it to the mapping we are building
            kwargs[param] = value
        return kwargs
    
    # just helps modularize load_from_config(...)
    @classmethod
    def load_broadcasters(cls, key, info, comp_name, model_name, broadcasters=None, verbose=True):

        # broadcasters specified by key 'broadcasters'
        if 'broadcasters' in info:        
            
            # for each function in broadcasters, add it to the function call stack
            # with any kwargs as necessary
            func_list = []
            kwarg_list = []
            for func_name, func_kwargs in info['broadcasters'].items():
                if broadcasters is not None: # for parameters
                    if verbose:
                        print(f'Appending {func_name} to broadcasting call stack for component {comp_name} (model {model_name}) param {key}')
                else: # for component
                    if verbose:
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
    def load_shape(cls, shapes, param, info, comp_name, model_name, verbose=True):
        # shapes specified by key 'shapes'
        if 'shape' in info:
            shape = info['shape']
            assert 'value' not in info, 'A shaped-config component cannot have a fixed value'
            if verbose:
                print(f'Setting component {comp_name} (model {model_name}) param {param} sampled shape to {shape}')
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
        
        method = config['interpolation']['method']
        order = method_order_key[method]

        # get span of each param(s) in the component
        model_name = component.model.__class__.__name__

        spans = {}
        for param in component.params:
            comp_block = config['interpolation'][model_name][param]
            low = comp_block['low']
            high = comp_block['high']
            N = comp_block['N']
            spans[param] = np.linspace(low, high, N) # follows params order
        # print(np.array(list(spans.values())).shape)

        # build interpolator
        nu = channel.bandpass.nu
        if len(spans) == 0:
            signal = component.model(nu)
            y = channel.bandpass.integrate_signal(signal) # this is one number!
            def interpolator(*args, **kwargs):
                return y
            interpolator_call_kwargs = {}

        elif len(spans) == 1:
            signal = component.model(nu, **spans)
            y = channel.bandpass.integrate_signal(signal)
            interpolator = interp.interp1d(*spans.values(), y, kind=method)
            interpolator_call_kwargs = {}

        elif len(spans) == 2:
            meshed_spans = np.meshgrid(*spans.values(), indexing='ij', sparse=True)
            meshed_spans = {k: v for k, v in zip(spans.keys(), meshed_spans)}
            signal = component.model(nu, **meshed_spans)
            y = channel.bandpass.integrate_signal(signal)
            interpolator = interp.RectBivariateSpline(*spans.values(), y, kx=order, ky=order)
            interpolator_call_kwargs = {'grid': False}

        else:
            raise NotImplementedError('Only up to 2-parameter models implemented so far')

        self.interpolator = interpolator
        self.interpolator_call_kwargs = interpolator_call_kwargs

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
        res = self.interpolator(*spans, **self.interpolator_call_kwargs)
        return self.channel.bandpass.rj_to_cmb * self.component.broadcaster(res)

class MixingMatrix:
    
    def __init__(self, channels, components, shape, wcs, dtype=np.float32):

        nchan = len(channels)
        ncomp = len(components)

        self._elements = {}
        self._dtype = dtype

        for comp in components:
            self._elements[comp.name] = []
            for chan in channels:
                self._elements[comp.name].append(Element(chan, comp))

        self._matrix = enmap.zeros((nchan, ncomp) + shape, wcs=wcs, dtype=dtype)

    def __call__(self, params_obj):
        assert params_obj.shape == self._matrix.shape[2:], \
            f'Params object shape {params_obj.shape} must equal matrix shape {self.matrix.shape[2:]}'
        
        # update Elements by component
        for compidx, (comp_name, active_params) in enumerate(params_obj.params.items()):
            for chanidx, element in enumerate(self._elements[comp_name]):
                self._matrix[chanidx, compidx] = element(**active_params)

        return self._matrix

    @classmethod
    def load_from_config(cls, config_path, verbose=True):
        try:
            config = utils.config_from_yaml_resource(config_path)
        except FileNotFoundError:
            config = utils.config_from_yaml_file(config_path)

        # get list of channels
        channels = []
        for instr, bands in config['channels'].items():
            for band, kwargs in bands.items():
                channels.append(data.Channel(instr, band, **kwargs))
                
        # get list of components
        components = []
        for comp_name in config['components']:
            components.append(Component.load_from_config(config_path, comp_name))

        # get pol, shape, wcs, dtype
        params_block = config['parameters']
        pol = params_block['pol']
        shape, wcs = enmap.read_map_geometry(params_block['geometry'])
        shape = (len(pol),) + shape[-2:]
        kwargs = {'dtype': params_block.get('dtype')} if params_block.get('dtype') else {}

        return cls(channels, components, shape, wcs, **kwargs)


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