from pixell import enmap
import numpy as np
import healpy as hp 

from tacos import constants as cs, units, utils, broadcasting

from abc import ABC, abstractmethod
from ast import literal_eval
import os


template_path = utils.data_dir_str('template')


class SED(ABC):

    # A sed simply needs to store its possible parameters as an interable property
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

class Dust(SED):
    
    def __init__(self, nu0=353e9, **kwargs):
        super().__init__(nu0, **kwargs)

    # define all possible params
    @property
    def params(self):
        return ['beta', 'T']

    def __call__(self, nu, **kwargs):
        return modified_blackbody_ratio(nu, self.nu0, **kwargs)

class Synch(SED):

    def __init__(self, nu0=30e9, **kwargs):
        super().__init__(nu0, **kwargs)

    # define all possible params
    @property
    def params(self):
        return ['beta']

    def __call__(self, nu, **kwargs):
        return power_law_ratio(nu, self.nu0, **kwargs)

class CMB(SED):

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

# RJ for now, so sed is power_law ** (beta + 1)
def modified_blackbody_ratio(nu, nu0, beta, T):
    nu0, beta, T = utils.expand_all_arg_dims(nu0, beta, T)
    x = nu * cs.hplanck() / (cs.kboltz() * T)
    x0 = nu0 * cs.hplanck() / (cs.kboltz() * T)
    return (nu / nu0) ** (beta + 1) * np.expm1(x0) / np.expm1(x) # RJ for now

# RJ for now, so sed is dw_dt(nu) * Delta_T_CMB
def thermodynamic_ratio(nu, nu0):
    return units.dw_dt(nu) / units.dw_dt(nu0)

### Component Wrapper Class ###

class Component:

    def __init__(self, sed, comp_name=None, fixed_params=None, param_broadcasters=None,
                comp_broadcaster=None, param_shapes=None, verbose=True):
        
        self.sed = sed
        self._name = comp_name if comp_name else sed.__class__.__name__.lower()

        # store any fixed params, label all else active params
        self._params = {
            'active': [],
            'fixed': {}
        }
        if fixed_params is None:
            fixed_params = {}
        for param in sed.params:
            val = fixed_params.get(param)
            if val:
                self.fixed_params[param] = val
            else:
                self.active_params.append(param)

        # store broadcasters. these are used to guarantee a parameter of a given 
        # shape will broadcast against the mixing matrix. there can be broadcasters
        # for any parameter, which are evaluated prior to interpolation, and a 
        # broadcaster for the entire component, which is evaluated after interpolation        

        # store per-parameter broadcasters, and evaluate them (once) for any fixed params
        self._param_broadcasters = {param: lambda x: x for param in sed.params}
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
        res = self.sed(nu, **kwargs)
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

        # get pixelization and dtype
        healpix = paramaters_block['healpix']
        try:
            dtype = paramaters_block['dtype']
        except KeyError:
            dtype = np.float32

        # first get the sed of the component
        sed_name = comp_block['sed']
        try:
            nu0 = comp_block['nu0']
            sed_kwargs = {'nu0': literal_eval(nu0)} # in case format is like 23e9
        except KeyError:
            # if not provided, use the default in models.py
            sed_kwargs = {}
        sed = getattr(__file__, sed_name)(verbose=verbose, **sed_kwargs)

        # get the component broadcasters, if any
        if 'broadcasters' in comp_block:
            comp_broadcaster = cls.parse_broadcasters(
                comp_block['broadcasters'], healpix,
                comp_name=comp_name, sed_name=sed_name, verbose=verbose
                )

        # get the possible fixed params, broadcasting function stack for each param, and shapes
        # of each param
        fixed_params = {}
        param_broadcasters = {}
        param_shapes = {}

        if 'params' in comp_block:
            for param_name, param_block in comp_block['params'].items():
                
                assert param_name in sed.params, f'Param {param_name} not in {sed_name} params'
                
                # if a particular param has a value, parse it
                if 'value' in param_block:
                    assert 'shape' not in param_block, 'A fixed value cannot have a config-set shape'
                    fixed_params[param_name] = cls.parse_value(
                        param_block['value'], healpix, dtype=dtype,
                        comp_name=comp_name, sed_name=sed_name, param_name=param_name, verbose=verbose
                        )

                # if a particular param has broadcaster(s), build the broadcasting function
                # call stack
                if 'broadcasters' in param_block:
                    param_broadcasters[param_name] = cls.parse_broadcasters(
                        param_block['broadcasters'], healpix,
                        comp_name=comp_name, sed_name=sed_name, param_name=param_name, verbose=verbose
                        )

                # if a particular param has a specific shape, load and store it
                if 'shape' in param_block:
                    assert 'value' not in param_block, 'A shaped-config component cannot have a fixed value'
                    param_shapes[param_name] = cls.parse_shape(
                        param_block['shape'],
                        comp_name=comp_name, sed_name=sed_name, param_name=param_name, verbose=verbose
                        )

        # get component
        return cls(sed, comp_name=comp_name, comp_broadcaster=comp_broadcaster,
                param_broadcasters=param_broadcasters, param_shapes=param_shapes, verbose=verbose)

    # just helps modularize load_from_config(...)
    @classmethod
    def parse_value(cls, value, healpix, dtype=np.float32,
                    comp_name=None, sed_name=None, param_name=None, verbose=True):
        scalar_verbose_str = f'Fixing component {comp_name} (model {sed_name}) param {param_name} to {float(value)}'
        fullpath_verbose_str = f'Fixing component {comp_name} (model {sed_name}) param {param_name} to data at {value}'
        resource_verbose_str = f'Fixing component {comp_name} (model {sed_name}) param {param_name} to {value} template'
        return utils.parse_maplike_value(
            value, healpix, template_path, dtype=dtype, scalar_verbose_str=scalar_verbose_str,
            fullpath_verbose_str=fullpath_verbose_str, resource_verbose_str=resource_verbose_str,
            verbose=verbose
            )
    
    # just helps modularize load_from_config(...)
    @classmethod
    def parse_broadcasters(cls, function_block, healpix,
        comp_name=None, sed_name=None, param_name=None, verbose=True):     
        # for each function in broadcasters, add it to the function call stack
        # with any kwargs as necessary
        func_list = []
        kwarg_list = []
        for func_name, func_kwargs in function_block.items():
            if verbose:
                msg = f'Appending {func_name} to broadcasting call stack for component {comp_name} (sed {sed_name})'
                if param_name:
                    msg += f' param {param_name}'
                print(msg)
            
            # append the function to the list
            broadcaster = getattr(broadcasting, func_name)
            func_list.append(broadcaster)

            # look for any kwargs, or append empty kwargs.
            # healpix a global parameter for the analysis, so handle it separately
            if (func_kwargs is None) or (func_kwargs == 'None'):
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
        comp_name=None, sed_name=None, param_name=None, verbose=True):        
        if verbose:
            print(f'Setting component {comp_name} (sed {sed_name}) param {param_name} sampled shape to {shape}')
        shape = literal_eval(shape) # this maps a stringified tuple to the actual tuple
        return shape

    @property
    def sed(self):
        return self._sed

    @property
    def name(self):
        return self._name

    @property
    def params(self):
        return self._sed.params

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