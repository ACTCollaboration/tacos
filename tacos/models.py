

# A generic class representing a component?

# Less generic class that represents a diffuse component (CMB, synch, dust etc.)

   # __init__ method that takes parameters: reference frequency..

   # __call__ method that takes array of frequencies and returns amplitudes (perhaps not needed)

   # Amplitude attribute? A map.. Will be updated inplace by Gibbs sampler. I, Q, U for example.

   # beta attribute, another map. Could be several parameters... beta_d and T for dust.. 

   # Perhaps, linear parameter attribute: dict of arrays. self.lin_par['amp']
   # and self.nonlin_par['beta_d'].

   # I.e the non-linear part
   # defined to be 1 at the reference frequency.
   
   # Total amplitude as function of frequency (amp + beta(nu)).
   
import numpy as np

from tacos import constants as cs, units, utils
from pixell import enmap

from abc import ABC, abstractmethod
import os

config = utils.config_from_yaml_resource('configs/models.yaml')

class DiffuseComponent(ABC):

    def __init__(self, name, nu0, params, **kwargs):
        self.nu0 = nu0
        self.name = name if name else self.__class__.__name__.lower()

        self.params = {
            'active': {},
            'fixed': {}
        }

        for param in params:
            val = kwargs.get(param)
            if val is not None:
                which = 'fixed'
            else:
                which = 'active'
            self.params[which][param] = val

    @classmethod
    def load_from_config(cls, config_path):

        # load the block under 'components' with name equal to the Class calling this method
        # first look in the default configs, then assume 'config' is a full path
        comp_class = cls.__name__
        try:
            comp_block = utils.config_from_yaml_resource(f'configs/{config_path}')['components'][comp_class]
        except FileNotFoundError:
            comp_block = utils.config_from_yaml_file(config_path)['components'][comp_class]

        kwargs = {}

        # get the name of the instance
        name = comp_block.get('name')
        if name: 
            kwargs['name'] = name

        # get the reference frequency
        nu0 = comp_block.get('nu0')
        if nu0:
            kwargs['nu0'] = name

        # for each parameter listed, determine if fixed and how to handle, or
        # if active, whether there is a broadcaster
        for param, info in comp_block.items():

            # fixed parameters specified by key 'value'
            if 'value' in info:
                assert len(info) == 1, 'Param {param} has fixed value {value}; no other information is allowed'

                # if scalar, fix parameter to that value everywhere 
                if isinstance(info, (int, float)):
                    print(f'Fixing {comp_class} param {param} to {float(info)}')
                    kwargs[param] = float(info)

                # if string, first see if it is in the config templates
                # if not, load it directly
                elif isinstance(info, str):
                    if info in config['templates']:
                        print(f'Fixing {comp_class} param {param} to {info} template')
                        kwargs[param] = enmap.read_map(config['templates'][info])
                    else:
                        print(f'Fixing {comp_class} param {param} to data at {info}')
                        kwargs[param] = enmap.read_map(info)
                
        # get component
        return cls(**kwargs)

    def get_all_params(self):
        res = self.get_fixed_params()
        res.update(self.get_active_params())
        return res

    def get_fixed_params(self):
        return self.params['fixed']
    
    def get_active_params(self):
        return self.params['active']
    
    @abstractmethod
    def __call__(self, nu):
        pass

class Dust(DiffuseComponent):
    
    def __init__(self, name='dust', nu0=353e9, **kwargs):
        
        # define all possible params
        params = ['beta', 'T']

        # perform operations common to DiffuseComponents
        super().__init__(name, nu0, params, **kwargs)

    def __call__(self, nu, **kwargs):
        kwargs.update(self.get_fixed_params())
        return modified_blackbody_ratio(nu, self.nu0, **kwargs)

class Synch(DiffuseComponent):

    def __init__(self, name='synch', nu0=30e9, **kwargs):

        # define all possible params
        params = ['beta']

        # perform operations common to DiffuseComponents
        super().__init__(name, nu0, params, **kwargs)

    def __call__(self, nu, **kwargs):
        kwargs.update(self.get_fixed_params())
        return power_law_ratio(nu, self.nu0, **kwargs)

class CMB(DiffuseComponent):

    def __init__(self, name='cmb', nu0=100e9, **kwargs):

        # define all possible params
        params = []

        # perform operations common to DiffuseComponents
        super().__init__(name, nu0, params, **kwargs)

    def __call__(self, nu, **kwargs):
        return thermodynamic_ratio(nu, self.nu0)

### SEDs ###

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
    nu0 = utils.expand_all_arg_dims(nu0)
    return units.dw_dt(nu) / units.dw_dt(nu0)
        