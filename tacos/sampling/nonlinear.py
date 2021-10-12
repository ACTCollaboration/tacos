import numpy as np

from tacos import utils

from abc import ABC, abstractmethod


class NonLinSampler(ABC):

    # implement non-trivial sampling strategies of non-linear parameters. they 
    # may be conditional on linear parameters, or may not be (ie, they sample
    # from the marginal distribution). there are many strategies and hence there
    # are diverse concrete subclasses

    def __init__(self, params, mixing_matrix, noise_models, data, prior_rvs=None, dtype=np.float32):

        # use scipy.stats.rv objects for priors?
        # prior_rvs would need to be a dict matching the mapping of params