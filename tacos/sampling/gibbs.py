import numpy as np
from scipy import interpolate as interp
import os

from pixell import enmap

from tacos import data, utils, component, config

class GibbsSampler:

    # Holds metadata about the sampling run, as well as the necessary objects: a Chain, 
    # MixingMatrix, LinSampler, and NonLinSampler (possibly)

    def __init__(self, chain, mixing_matrix, linsampler=None, nonlinsampler=None, init_amplitudes=None,
                 num_steps=1000, dtype=np.float32):
        self._chain = chain
        self._mixing_matrix = mixing_matrix
        self._linsampler = linsampler
        self._nonlinsampler = nonlinsampler
        
        self._num_steps = num_steps
        self._dtype = dtype

    def step(self):
        
        # somehow we need a starting point, whether or not beta, and whether or not priors
        # do we always start with a beta sample, if available? I think so

        # this is more pseudo-code than real code
        if self._nonlinsampler:
            pass

    def run(self, step_per_save=None):
        pass

    @classmethod
    def load_from_config(cls):
        pass