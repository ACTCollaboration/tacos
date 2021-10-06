from pixell import enmap
import numpy as np
import healpy as hp 

from tacos import constants as cs, units, utils, broadcasting

from abc import ABC, abstractmethod
from ast import literal_eval
import os


class LinSampler(ABC):

    # a linear sampler depends on a tacos.tacos.MixingMatrix instance and an
    # mnms.mnms.NoiseModel instance to work. It also may take a prior, which 
    # is also an mnms.mnms.NoiseModel instance

    # TODO: some data channels might be correlated, and therefore share a noise model!

    # TODO: using an mnms.mnms.NoiseModel instance presumes we are in the 
    # pixel basis. Therefore a "SingleBasis" sampler is necessarily a pixel-diagonal
    # sampler, as opposed to eg a harmonic basis sampler (possibly). Maybe this
    # can be generalized? If so, to what: harmonic, tiled, wavelet bases?
    
    # TODO: is it possible to have a mixing matrix and beam be exactly represented
    # in a harmonic, tiled, or wavelet basis?
    #
    # >>>I think a mixing matrix might be well-defined in harmonic space, which 
    # would eliminate one alm2map transform. If transform data to harmonic space,
    # then eliminates all SHTs associated with beam, and only leaves those in
    # noise model. There are more alms than pixels though. Seljebotn states that
    # it's best to do this still in the pixel domain but using GL pixels...
    #
    # >>>I think a beam function B_ell might be well-defined as a special case of
    # the last two bases (where the tiled or wavelet power is unity, and B_ell
    # given in global harmonic filter)

    def __init__(self, mixing_matrix, noise_models, data, prior_models=None, priors=None, dtype=np.float32):
        self._mixing_matrix = mixing_matrix
        self._noise_models = noise_models
        self._prior_models = prior_models
        self._dtype = dtype

        num_chan, num_comp = self._mixing_matrix.shape[:2]
        self._num_chan = num_chan
        self._num_comp = num_comp

        # save noise-filtered data, which we only need to calculate once
        # has shape (num_chan, num_pol, ...)
        self._Ninvd = np.array([noise_models[i].filter(data[i]) for i in range(num_chan)], dtype=dtype)

        # if prior, save prior-filtered priors, which we only need to calculate once
        if priors is not None:
            assert prior_models, 'Supplied priors with prior_models, this is not allowed'
            self._Sinvm = np.array([prior_models[i].filter(priors[i]) for i in range(num_comp)], dtype=dtype)
        else:
            self._Sinvm = None
        
        assert len(noise_models) == num_chan, \
            f'noise_models must be an iterable over {num_chan} elements, got {len(noise_models)} instead'
        assert len(data) == num_chan, \
            f'data must be an iterable over {num_chan} elements, got {len(data)} instead'
        if prior_models:
            assert len(prior_models) == num_comp, \
                f'prior_models must be an iterable over {num_comp} elements, got {len(prior_models)} instead'
        if priors:
            assert len(priors) == num_comp, \
                f'priors must be an iterable over {num_comp} elements, got {len(priors)} instead'
        assert mixing_matrix.dtype == dtype, \
            f'Mixing matrix dtype {mixing_matrix.dtype} does not match provided dtype {self._dtype}'

    def __call__(self, noise_seed=None, prior_seed=None, M=None, chain=None, iteration=-1, **comp_params):
        # get the mixing matrix for this call, if not provided
        if not M:
            M = self._mixing_matrix(chain=chain, iteration=iteration, **comp_params)
        
        # get random samples for noise and prior (if any)
        eta_d = utils.concurrent_standard_normal(
            size=(self._num_chan, *self._mixing_matrix.element_shape), seed=noise_seed, dtype=self._dtype
            )

        if self._prior_models:
            eta_d = utils.concurrent_standard_normal(
                size=(self._num_comp, *self._mixing_matrix.element_shape), seed=prior_seed, dtype=self._dtype
                )
        else:
            eta_s = None

        # build RHS
        RHS = self._get_RHS(M, eta_d, eta_s=eta_s)

        # solve for linear sample. M needed to define 'A', while 'RHS' is b
        return self._solve(M, RHS)

    def _get_RHS(self, M, eta_d, eta_s=None):
        # TODO: implement correlated filtering

        # want to do this MN_invd = np.einsum('jca...,jab...,jb...->ca...', M, N_inv, d)
        # which is the same as the below
        # NOTE: because of the intermediate calculation, you double the floating point error
        # so e.g. 32bit float 1e-6 -> 1e-5 vs 64bit float 1e-15->1e-14

        # TODO: parallelize einsums
        RHS = np.einsum('jc...,j...->c...', M, self._Ninvd)

        # avoid this sum if Sinvm is 0
        if self._Sinvm:
            RHS += self._Sinvm
        
        # get the noise sample
        eta_d = np.array(
            [self._noise_models[i].filter(eta_d[i], power=-0.5) for i in range(self._num_chan)], dtype=self._dtype
            )
        RHS += np.einsum('jc...,j...->c...', M, eta_d)

        # avoid prior sample if not necessary
        if eta_s:
            eta_s = np.array(
                [self._prior_models[i].filter(eta_s[i], power=-0.5) for i in range(self._num_comp)], dtype=self._dtype
                )
            RHS += eta_s

        return RHS

    @abstractmethod
    def _solve(self, M, RHS):
        pass
          

class SingleBasis(LinSampler):

    # Take advantage of all objects being in the same basis to perform an "exact" inversion
    # of the amplitude sampling. This assumes of course that all objects are sufficiently
    # sparse even in this basis. Also makes same pixel-space assumption as base class (for now)

    def __init__(self, mixing_matrix, noise_models, data, prior_models=None, priors=None, dtype=np.float32):
        super().__init__(mixing_matrix, noise_models, data, prior_models=prior_models,
                            priors=priors, dtype=dtype)

        # we want to precompute N^-1, S^-1, N^-0.5, and S^-0.5 such that the NoiseModel
        # instances are guaranteed to hold a reference to these matrices explicitly. That
        # way, a call to filter(), filter(power=-0.5) is fast, as is the call to
        # NoiseModel.model to build the LHS

    def _solve(self, M, RHS):
        pass

class MixedBasis(LinSampler):

    # This must solve the sampling equation iteratively, assuming the basis transformations will
    # make objects very dense if represented in one basis. 

    def _solve(self, M, RHS):
        # CG solver!
        pass