import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid as cumtrapz

from tacos import utils

from abc import ABC, abstractmethod
import time


module_config = utils.config_from_yaml_resource('configs/nonlinear.yaml')

method_order_key = {
    'linear': 1,
    'quadratic': 2,
    'cubic': 3
} 

def clock(ref):
    now = time.time()
    print(now - ref)
    return now

class NonLinSampler(ABC):

    # implement non-trivial sampling strategies of non-linear parameters. they 
    # may be conditional on linear parameters, or may not be (ie, they sample
    # from the marginal distribution). there are many strategies and hence there
    # are diverse concrete subclasses

    def __init__(self, mixing_matrix, noise_models, data, comp, param, prior_rv=None, dtype=None):
        self._mixing_matrix = mixing_matrix
        self._noise_models = noise_models
        self._comp = comp
        self._param = param
        self._prior_rv = prior_rv
        self._dtype = dtype if dtype else np.float32

        self._data = np.asarray(data, dtype=self._dtype)

        num_chan, num_comp, num_pol = self._mixing_matrix.shape[:3]
        self._num_chan = num_chan
        self._num_comp = num_comp
        self._num_pol = num_pol

        # use scipy.stats.rv objects for priors?
        # prior_rvs would need to be a dict matching the mapping of params
        pass

    def _log_like(self, params, a):
        # only want to update elements with
        # comp params that have been passed, so that successive calls
        # are as fast as possible
        M = self._mixing_matrix(**{self._comp: params[self._comp]})
                
        n = np.einsum('jca...,ca...->ja...', M, a)        
        n = self._data - n
        
        Ninvn = np.array(
            [self._noise_models[i].filter(n[i]) for i in range(self._num_chan)],
            dtype=self._dtype
            )
        
        chi2 = np.einsum('i,i->', n.reshape(-1), Ninvn.reshape(-1))
        return -0.5 * chi2

    def _post(self, params, a):
        
        post = np.exp(self._log_like(params, a))

        if self._prior_rv is not None:
            param = params[self._comp][self._param]
            post *= self._prior_rv.pdf(param.reshape(-1))

        return post

    @abstractmethod
    def __call__(self, amplitudes, params, seed=None):
        pass

class InversionSampler(NonLinSampler):

    def __init__(self, mixing_matrix, noise_models, data, comp, param, prior_rv=None, dtype=None):
        super().__init__(mixing_matrix, noise_models, data, comp, param,
                            prior_rv=prior_rv, dtype=dtype)
        
        # get inversion 1d grid and interpolation order
        method = module_config['inversion_sampler']['method']
        self._order = method_order_key[method]
        
        comp_block = module_config['inversion_sampler'][comp][param]
        self._low = comp_block['low']
        self._high = comp_block['high']
        self._coarse_N = comp_block['coarse_N']
        self._fine_N = comp_block['fine_N']
        self._coarse_grid = np.linspace(self._low, self._high, self._coarse_N)

    def _sample_beta_pix(self, idx, amplitudes, params, seed=None, size=1):
        # get gridded values
        param = params[self._comp][self._param]
        coarse_logpdf = np.empty(self._coarse_grid.size, self._dtype)
        for i, coarse_grid_val in enumerate(self._coarse_grid):
            param[idx] = coarse_grid_val
            params[self._comp][self._param] = param
            coarse_logpdf[i] = self._log_like(params, amplitudes)

        # make more numerically stable by normalizing to the max logpdf
        coarse_logpdf -= np.max(coarse_logpdf)
        coarse_pdf = np.exp(coarse_logpdf)

        # find the fine boundary
        coarse_logpdf_interp = interp1d(self._coarse_grid, coarse_logpdf, kind='cubic')
        fine_grid = np.linspace(self._low, self._high, 1_000_000)
        fine_logpdf = coarse_logpdf_interp(fine_grid)
        fine_grid = fine_grid[fine_logpdf >= -12.5]
        self._fine_grid = np.linspace(fine_grid[0], fine_grid[-1], self._fine_N)

        # get fine gridded values
        param = params[self._comp][self._param]
        fine_logpdf = np.empty(self._fine_grid.size, self._dtype)
        for i, fine_grid_val in enumerate(self._fine_grid):
            param[idx] = fine_grid_val
            params[self._comp][self._param] = param
            fine_logpdf[i] = self._log_like(params, amplitudes)

        # make more numerically stable by normalizing to the max logpdf
        fine_logpdf -= np.max(fine_logpdf)
        fine_pdf = np.exp(fine_logpdf)

        # return coarse_logpdf, coarse_pdf, fine_logpdf, fine_pdf

        # construct ppf, rescale C to 0..1 range
        c = cumtrapz(fine_pdf, x=self._grid, initial=0)
        c = (c-c[0])/(c[-1]-c[0])
        _, uidx = np.unique(c, return_index=True)
        ppf = interp1d(c[uidx], self._grid[uidx], kind=self._order)

        # get random sample
        rng = np.random.default_rng(seed)
        return ppf(rng.random(size=size))

    def __call__(self, amplitudes, params, seed=None, size=1):
        param = params[self._comp][self._param]

        for idx in np.ndindex(param.shape):
            param[idx] = self._sample_beta_pix(idx, amplitudes, params, seed=seed, size=size)
            params[self._comp][self._param] = param

        return params
