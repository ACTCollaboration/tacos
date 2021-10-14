import numpy as np

from tacos import constants as cs, units, utils

from abc import ABC, abstractmethod

REGISTERED_SEDS = {}


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

@utils.register(REGISTERED_SEDS)
class Dust(SED):
    
    def __init__(self, nu0=353e9, **kwargs):
        super().__init__(nu0, **kwargs)

    # define all possible params
    @property
    def params(self):
        return ['beta', 'T']

    def __call__(self, nu, **kwargs):
        return modified_blackbody_ratio(nu, self.nu0, **kwargs)

@utils.register(REGISTERED_SEDS)
class Synch(SED):

    def __init__(self, nu0=30e9, **kwargs):
        super().__init__(nu0, **kwargs)

    # define all possible params
    @property
    def params(self):
        return ['beta']

    def __call__(self, nu, **kwargs):
        return power_law_ratio(nu, self.nu0, **kwargs)

@utils.register(REGISTERED_SEDS)
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