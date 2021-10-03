from pixell import enmap
import numpy as np
import healpy as hp 

from tacos import constants as cs, units, utils, broadcasting

from abc import ABC, abstractmethod
from ast import literal_eval
import os


template_path = utils.data_dir_str('template')


class Model(ABC):

    # A model simply needs to store its possible parameters as an interable property
    # and be callable over frequencies

    def __init__(self, nu0, verbose=True):
        if verbose:
            print(f'Setting {self.__class__.__name__} reference frequency to {nu0/1e9} GHz')
        self._nu0 = nu0

    @property
    def nu0(self):
        return self._nu0

    @property
    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def __call__(self, nu, **kwargs):
        pass

class Dust(Model):
    
    def __init__(self, nu0=353e9, **kwargs):
        super().__init__(nu0, **kwargs)

    # define all possible params
    @property
    def params(self):
        return ['beta', 'T']

    def __call__(self, nu, **kwargs):
        return modified_blackbody_ratio(nu, self.nu0, **kwargs)

class Synch(Model):

    def __init__(self, nu0=30e9, **kwargs):
        super().__init__(nu0, **kwargs)

    # define all possible params
    @property
    def params(self):
        return ['beta']

    def __call__(self, nu, **kwargs):
        return power_law_ratio(nu, self.nu0, **kwargs)

class CMB(Model):

    def __init__(self, nu0=100e9, **kwargs):
        super().__init__(nu0, **kwargs)

    # define all possible params
    @property
    def params(self):
        return []

    def __call__(self, nu):
        return thermodynamic_ratio(nu, self.nu0)

### SEDs ###

# frequencies along -1 axis, so must first append a new axis to each argument of each SED

def power_law_ratio(nu, nu0, beta):
    nu0, beta = utils.expand_all_arg_dims(nu0, beta)
    return (nu / nu0) ** beta

# RJ for now, so model is power_law ** (beta + 1)
def modified_blackbody_ratio(nu, nu0, beta, T):
    nu0, beta, T = utils.expand_all_arg_dims(nu0, beta, T)
    x = nu * cs.hplanck() / (cs.kboltz() * T)
    x0 = nu0 * cs.hplanck() / (cs.kboltz() * T)
    return (nu / nu0) ** (beta + 1) * np.expm1(x0) / np.expm1(x) # RJ for now

# RJ for now, so model is dw_dt(nu) * Delta_T_CMB
def thermodynamic_ratio(nu, nu0):
    return units.dw_dt(nu) / units.dw_dt(nu0)

### Component Wrapper Class ###

class Component:

    def __init__(self, model, comp_name=None, fixed_params=None, param_broadcasters=None,
                comp_broadcaster=None, param_shapes=None, verbose=True):
        
        self._model = model
        self._name = comp_name if comp_name else model.__class__.__name__.lower()

        # store any fixed params, label all else active params
        self._params = {
            'active': [],
            'fixed': {}
        }
        if fixed_params is None:
            fixed_params = {}
        for param in model.params:
            val = fixed_params.get(param)
            if val:
                self.fixed_params[param] = val
            else:
                self.active_params.append(param)

        # store broadcasters. these are used to guarantee a parameter of a given 
        # shape will broadcast against the amplitude map. there can be broadcasters
        # for any parameter, which are evaluated prior to interpolation, and a 
        # broadcaster for the entire component, which is evaluated after interpolation        

        # store per-parameter broadcasters, and evaluate them (once) for any fixed params
        self._param_broadcasters = {param: lambda x: x for param in model.params}
        if param_broadcasters is None:
            param_broadcasters = {}
        for param in param_broadcasters:
            if param in self.fixed_params:
                if verbose: 
                    print(f'Broadcasting fixed param {param} and overwriting its value to ' + \
                        'broadcasted result')
                self.fixed_params[param] = param_broadcasters[param](self.fixed_params[param])
            elif param in self.active_params:
                self.param_broadcasters[param] = param_broadcasters[param]
            else:
                raise ValueError(f'Param {param} not in fixed nor active params')
        
        # store component-level broadcaster
        self._comp_broadcaster = lambda x: x
        if comp_broadcaster is not None:
            self._comp_broadcaster = comp_broadcaster

        # store shape for passing onto Chain class, but it doesn't do anything here
        self._param_shapes = {}
        if param_shapes is None:
            param_shapes = {}
        for param, shape in param_shapes.items():
            if param in self.active_params:
                self.param_shapes[param] = shape

    # This function oddly has no use when things are interpolated
    # I think it will come in handy when evaluating a proposal that has gone "out of bounds"
    # TODO: implement that
    def __call__(self, nu, **kwargs):
        
        # first broadcast active params
        for param in self.active_params:
            kwargs[param] = self.param_broadcasters[param](kwargs[param])

        # then add in the already-broadcasted fixed params
        for param in self.fixed_params:
            assert param not in kwargs, f'Param {param} is fixed but was passed as a kwarg'
        kwargs.update(self.fixed_params)

        # finally, evaluate
        res = self.model(nu, **kwargs)
        return self.comp_broadcaster(res)

    @classmethod
    def load_from_config(cls, config_path, comp_name, verbose=True):

        # first look in the default configs, then assume 'config' is a full path
        try:
            config = utils.config_from_yaml_resource(config_path)
        except FileNotFoundError:
            config = utils.config_from_yaml_file(config_path)

        paramaters_block = config['parameters']
        comp_block = config['components'][comp_name]

        # get pixelization
        healpix = paramaters_block['healpix']

        # first get the model of the component
        model_name = comp_block['model']
        nu0 = comp_block.get('nu0') # if not provided, use the default in models.py
        model_kwargs = {'nu0': literal_eval(nu0)} if nu0 else {}
        model = getattr(__file__, model_name)(verbose=verbose, **model_kwargs)

        # get the component broadcasters, if any
        if 'broadcasters' in comp_block:
            functions_block = comp_block['broadcasters']
            comp_broadcaster = cls.parse_broadcasters(
                functions_block, healpix,
                comp_name=comp_name, model_name=model_name, verbose=verbose
                )

        # get the possible fixed params, broadcasting function stack for each param, and shapes
        # of each param
        fixed_params = {}
        param_broadcasters = {}
        param_shapes = {}

        if 'params' in comp_block:
            for param_name, param_block in comp_block['params'].items():
                
                assert param_name in model.params, f'Param {param_name} not in {model_name} params'
                
                # if a particular param has a value, parse it
                if 'value' in param_block:
                    assert 'shape' not in param_block, 'A fixed value cannot have a config-set shape'
                    value = param_block['value']
                    value = cls.parse_value(
                        value, healpix,
                        comp_name=comp_name, model_name=model_name, param_name=param_name, verbose=verbose
                        )
                    fixed_params[param_name] = value

                # if a particular param has broadcaster(s), build the broadcasting function
                # call stack
                if 'broadcasters' in param_block:
                    functions_block = param_block['broadcasters']
                    stacked_broadcaster = cls.parse_broadcasters(
                        functions_block, healpix,
                        comp_name=comp_name, model_name=model_name, param_name=param_name, verbose=verbose
                        )
                    param_broadcasters[param_name] = stacked_broadcaster

                # if a particular param has a specific shape, load and store it
                if 'shape' in param_block:
                    assert 'value' not in param_block, 'A shaped-config component cannot have a fixed value'
                    shape = param_block['shape']
                    shape = cls.parse_shape(
                        shape,
                        comp_name=comp_name, model_name=model_name, param_name=param_name, verbose=verbose
                        )
                    param_shapes[param_name] = shape

        # get component
        return cls(model, comp_name=comp_name, comp_broadcaster=comp_broadcaster,
                param_broadcasters=param_broadcasters, param_shapes=param_shapes, verbose=verbose)

    # just helps modularize load_from_config(...)
    @classmethod
    def parse_value(cls, value, healpix,
        comp_name=None, model_name=None, param_name=None, verbose=True):
        # if scalar, fix parameter to that value everywhere 
        if isinstance(value, (int, float)):
            if verbose:
                print(f'Fixing {model_name} param {param_name} to {float(value)}')
            value = float(value)

        # if string, first see if it exists as a fullpath to a file. if so, load it directly
        elif isinstance(value, str):
            if os.path.exists(value):
                if verbose:
                    print(f'Fixing component {comp_name} (model {model_name}) param {param_name} to data at {value}')
                if healpix:
                    value = hp.read_map(value, field=None, dtype=np.float32)
                else:
                    value = enmap.read_map(value)
            
            # if not fullpath, try loading it as a preexisting template      
            else:
                template_fullpath = template_path + value + '.' + utils.extensions['template']
                if verbose:
                    print(f'Fixing component {comp_name} (model {model_name}) param {param_name} to {value} template')
                if healpix:
                    value += '_healpix'
                    value = hp.read_map(template_fullpath, field=None, dtype=np.float32)
                else:
                    value = enmap.read_map(template_fullpath)
        
        return value
    
    # just helps modularize load_from_config(...)
    @classmethod
    def parse_broadcasters(cls, function_block, healpix,
        comp_name=None, model_name=None, param_name=None, verbose=True):     
        # for each function in broadcasters, add it to the function call stack
        # with any kwargs as necessary
        func_list = []
        kwarg_list = []
        for func_name, func_kwargs in function_block.items():
            if verbose:
                msg = f'Appending {func_name} to broadcasting call stack for component {comp_name} (model {model_name})'
                if param_name:
                    msg += f' param {param_name}'
                print(msg)
            
            broadcaster = getattr(broadcasting, func_name)
            func_list.append(broadcaster) 
            if func_kwargs == 'None':
                kwarg_list.append({'healpix': healpix})
            else:
                func_kwargs.update({'healpix': healpix})
                kwarg_list.append(func_kwargs)

        # build a single function call stack
        def stacked_broadcaster(x):
            for i in range(len(func_list)):
                x = func_list[i](x, **kwarg_list[i])
            return x
        
        return stacked_broadcaster
    
    # just helps modularize load_from_config(...)
    @classmethod
    def parse_shape(cls, shape,
        comp_name=None, model_name=None, param_name=None, verbose=True):        
        if verbose:
            print(f'Setting component {comp_name} (model {model_name}) param {param_name} sampled shape to {shape}')
        shape = literal_eval(shape) # this maps a stringified tuple to the actual tuple
        return shape

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
    def comp_broadcaster(self):
        return self._comp_broadcaster

    @property
    def param_broadcasters(self):
        return self._param_broadcasters

    @property
    def param_shapes(self):
        return self._param_shapes