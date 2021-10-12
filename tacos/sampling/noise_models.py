import numpy as np
from mnms import noise_models

from tacos import utils

# this is a wrapper module to help expose available noise models to tacos, both
# those that are defined in mnms and defined within tacos, without dealing
# with messy Python namespace stuff

# every model here must be a Gaussian noise model that exposes a "filter" method,
# which weights a data vector (pixel-space map) by the model to an arbitrary
# power, specifically at least -1 and -1/2

REGISTERED_NOISE_MODELS = noise_models.REGISTERED_NOISE_MODELS

template_path = utils.data_dir_str('template')


@noise_models.register(registry=REGISTERED_NOISE_MODELS)
class SimplePixelNoiseModel:

    def __init__(self, inv_cov_mat, shape=None, dtype=None, polstr=None, cov_mult_fact=None):
        
        # cast inv_cov_mat to array of correct dtype and shape
        inv_cov_mat = np.asarray(inv_cov_mat, dtype=dtype)
        assert inv_cov_mat.ndim == 4, \
            'Inverse covariance for a single dataset/prior must have shape (npol, npol, ny, nx)'
        
        # post-process: slice out pol indices, multiply by cov_mult_factor
        self._polidxs = utils.polstr2polidxs(polstr)
        if shape is not None:
            shape = (len(self._polidxs), len(self._polidxs), *shape[-2:])
            inv_cov_mat = np.broadcast_to(inv_cov_mat, shape, subok=True)
        else:
            inv_cov_mat = inv_cov_mat[np.ix_(self._polidxs, self._polidxs)]
        if cov_mult_fact is None:
            cov_mult_fact = 1.
        inv_cov_mat /= cov_mult_fact # cov_mult_fact applies to covariance!
        
        self._nm_dict = {'inv_cov_mat': inv_cov_mat} # to be consistent with mnms
        self._shape = self.model.shape
        self._dtype = dtype if dtype else np.float32
        self._cov_mult_fact = cov_mult_fact

    def get_sim(self, split_num, sim_num, *str2seed_args):
        seed = (split_num, sim_num)
        for arg in str2seed_args:
            seed += (utils.hash_str(arg),)

        eta = utils.concurrent_standard_normal(
            size=(self._shape[1:]), seed=seed, dtype=self._dtype
            )
        return self.filter(eta, power=0.5)

    def filter(self, imap, power=-1):
        assert imap.shape == self._shape[1:], \
            f'Covariance matrix has shape {self._shape}, so imap must have shape {self._shape[1:]}'
        if power == -1:
            model = self.model
        else:
            try:
                # so we don't have to recompute if filtering by the same power later
                model = self._nm_dict[power]
            except KeyError:
                # because self.model already has -1 in exponent
                model = utils.eigpow(self.model, -power, axes=[-4, -3])
                self._nm_dict[power] = model
        return np.einsum('ab...,b...->a...', model, imap)

    @classmethod
    def load_from_value(cls, value, healpix, dtype=None, verbose=True):
        dtype = dtype if dtype else np.float32

        scalar_verbose_str = f'Fixing {cls.__name__} to {float(value)}'
        fullpath_verbose_str = f'Fixing {cls.__name__} to data at {value}'
        resource_verbose_str = f'Fixing {cls.__name__} to {value} template'
        return utils.parse_maplike_value(
            value, healpix, template_path, dtype=dtype, scalar_verbose_str=scalar_verbose_str,
            fullpath_verbose_str=fullpath_verbose_str, resource_verbose_str=resource_verbose_str,
            verbose=verbose
            )
            
    @property
    def model(self):
        return self._nm_dict['inv_cov_mat']

    @model.setter
    def model(self, value):
        # anything that can broadcast is OK, so do icov[:]= instead of icov=
        self._nm_dict['inv_cov_mat'][:] = value

