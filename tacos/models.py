

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

class DiffuseComponent(ABC):
    
    @abstractmethod
    def compute_amplitude(nu):
        
        pass

    @abstractmethod
    def compute_interpolation(self):
        
        # Given range of betas, compute interpolation lookup table.
        # For single beta: 1d table, for two betas 2d table etc.
        
        raise NotImplementedError()

class Dust(DiffuseComponent):
    
    def __init__(self):
        pass

class NonLinPar(ABC):
    
    @abstractmethod
    def __call__(self, nu):
        
        
