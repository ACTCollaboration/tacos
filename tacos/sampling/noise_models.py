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

    # TODO: handle more than polarization correlations? ie, channel correlations

    def __init__(self, inv_cov_mat, shape=None, dtype=None, polstr=None, cov_mult_fact=None):
        dtype = dtype if dtype else np.float32

        # cast inv_cov_mat to array of correct dtype and shape
        inv_cov_mat = np.asarray(inv_cov_mat, dtype=dtype)
        if shape is not None:
            inv_cov_mat = self._massage_input(inv_cov_mat, shape[-2:])
        
        # slice out pol dimensions
        self._polidxs = utils.polstr2polidxs(polstr)
        inv_cov_mat = inv_cov_mat[np.ix_(self._polidxs, self._polidxs)]
        assert inv_cov_mat.ndim == 4, \
            'Inverse covariance for a single dataset/prior must have shape (npol, npol, ny, nx)'    
        
        # scale covariance by mult_fact if provided
        if cov_mult_fact is None:
            cov_mult_fact = 1.
        inv_cov_mat /= cov_mult_fact # cov_mult_fact applies to covariance!
        
        self._nm_dict = {'inv_cov_mat': inv_cov_mat} # to be consistent with mnms
        self._shape = self.model.shape
        self._dtype = dtype if dtype else np.float32
        self._cov_mult_fact = cov_mult_fact

    def _massage_input(self, arr, shape):
        """Put input array into correct shape, which is (3, 3, ny, nx). If
        arr.ndim is not 4, fill the diagonals of the output with arr.

        Parameters
        ----------
        arr : array-like or scalar
            Input to be reshaped.
        shape : iterable
            Length-2 iterable containing (ny, nx). Actively used if arr is
            scalar or 1-d. Checks for compatibility if arr is >1-d.

        Returns
        -------
        array-like
            Shape (3, 3, ny, nx) representation of input, with input along
            axes=(0,1) diagonal if input dimension is less than 4.

        Raises
        ------
        ValueError
            If input dimension is greater than 4.
        """
        assert len(shape) == 2
        arr = np.atleast_1d(arr)
        indim = arr.ndim
        if indim > 4:
            raise ValueError(f'Input array has dimension {indim}, expected <= 4')
        elif indim == 1:
            if arr.size > 1:
                # assume along main diagonal
                arr = arr.reshape(arr.size, 1, 1)
                arr = np.broadcast_to(arr, (arr.size, *shape), subok=True)
            else:
                # scalar
                arr = np.broadcast_to(arr, shape, subok=True)
        assert arr.shape[-2:] == shape, \
            f'Massaged array has (ny, nx) = {arr.shape[-2:]}, expected {shape}'

        if indim == 4:
            assert arr.shape[:2] == (3, 3), \
                f'Massaged array has (npol, npol) = {arr.shape[:2]}, expected (3, 3)'
            return arr
        else:
            # must broadcast in first dimension if arr.ndim == 3!
            return np.eye(3, dtype=arr.dtype)[..., None, None] * arr

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

