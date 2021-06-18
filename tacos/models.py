

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
from abc import ABC, abstractmethod

from tacos import constants as cs

class DiffuseComponent(ABC):
    
    @abstractmethod
    def __call__(nu):
        pass

    # @abstractmethod
    # def compute_interpolation(self):
        
    #     # Given range of betas, compute interpolation lookup table.
    #     # For single beta: 1d table, for two betas 2d table etc.
        
    #     raise NotImplementedError()

class Dust(DiffuseComponent):
    
    def __init__(self, nu0, beta, T):
        # frequencies along -1 axis
        self.nu0 = np.expand_dims(np.atleast_1d(nu0), -1) 
        self.beta = np.expand_dims(np.atleast_1d(beta) , -1) 
        self.T = np.expand_dims(np.atleast_1d(T), -1) 

    def __call__(self, nu):
        return modified_blackbody_ratio(nu, self.nu0, self.beta, self.T)

class Synch(DiffuseComponent):

    def __init__(self, nu0, beta):
        # frequencies along -1 axis
        self.nu0 = np.expand_dims(np.atleast_1d(nu0), -1)
        self.beta = np.expand_dims(np.atleast_1d(beta), -1)

    def __call__(self, nu):
        return power_law_ratio(nu, self.nu0, self.beta)

def power_law_ratio(nu, nu0, beta):
    return (nu / nu0) ** beta

# RJ for now, so model is power_law ** (beta + 1)
def modified_blackbody_ratio(nu, nu0, beta, T):
    x = nu * cs.hplanck() / (cs.kboltz() * T)
    x0 = nu0 * cs.hplanck() / (cs.kboltz() * T)
    return power_law_ratio(nu, nu0, beta + 1) * np.expm1(x0) / np.expm1(x) # RJ for now

class NonLinPar(ABC):
    
    @abstractmethod
    def __call__(self, nu):
        pass
        