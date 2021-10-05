from pixell import enmap
import numpy as np
import healpy as hp 

from tacos import constants as cs, units, utils, broadcasting

from abc import ABC, abstractmethod
from ast import literal_eval
import os


class LinSampler(ABC):

    # a linear sampler depends on a tacos.tacos.MixingMatrix instance and an
    # mnms.mnms.NoiseModel instance to work. It also may take a prior instance

    def __init__(self, mixing_matrix, noise_model):
        pass