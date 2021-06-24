

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

class Model(ABC):

    # A model simply needs to store its possible parameters as an interable property
    # and be callable over frequencies

    def __init__(self, nu0):
        print(f'Setting reference frequency to {nu0/1e9} GHz')
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
    
    def __init__(self, nu0=353e9):
        super().__init__(nu0)

    # define all possible params
    @property
    def params(self):
        return ['beta', 'T']

    def __call__(self, nu, **kwargs):
        return modified_blackbody_ratio(nu, self.nu0, **kwargs)

class Synch(Model):

    def __init__(self, nu0=30e9):
        super().__init__(nu0)

    # define all possible params
    @property
    def params(self):
        return ['beta']

    def __call__(self, nu, **kwargs):
        return power_law_ratio(nu, self.nu0, **kwargs)

class CMB(Model):

    def __init__(self, nu0=100e9):
        super().__init__(nu0)

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