from pixell import enmap
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

@noise_models.register(registry=REGISTERED_NOISE_MODELS)
class SimplePixelNoiseModel:

    def __init__(self, inv_cov_mat, shape=None, dtype=np.float32, polstr=None, mult_fact=1):
        
        # cast inv_cov_mat to array of correct dtype and shape
        inv_cov_mat = np.asarray(inv_cov_mat, dtype=dtype)
        if shape is not None:
            inv_cov_mat = np.broadcast_to(inv_cov_mat, shape)
        assert inv_cov_mat.ndim == 4, \
            'Inverse covariance for a single dataset/prior must have shape (npol, npol, ny, nx)'
        
        # post-process: slice out pol indices, multiply by mult_factor
        self._polidxs = utils.polstr2polidxs(polstr)
        inv_cov_mat = inv_cov_mat[np.ix_(self._polidxs, self._polidxs)]
        inv_cov_mat /= mult_fact # mult_fact applies to covariance!
        
        self._nm_dict = {'inv_cov_mat': inv_cov_mat}
        self._shape = self.model.shape
        self._dtype = dtype
        self._mult_fact = mult_fact

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
    def load_from_channel(cls, channel, polstr=None, mult_fact=1):
        instr = channel.instr
        band = channel.band
        id = channel.id
        set = channel.set
        notes = channel.notes

        assert set in ['set0', 'set1', 'coadd']
        inv_cov_mat_path = utils.data_dir_str('covmat', instr)
        if not notes:
            notes = ''
        else:
            notes = '_' + notes
        inv_cov_mat_path += utils.data_fn_str(
            type='icovar', instr=instr, band=band, id=id, set=set, notes=notes
            )
        
        inv_cov_mat = enmap.read_map(inv_cov_mat_path)
        return cls(inv_cov_mat, dtype=inv_cov_mat.dtype, polstr=polstr, mult_fact=mult_fact)

    @classmethod
    def load_from_config(cls, config_path, comp_name, verbose=True):

        # first look in the default configs, then assume 'config' is a full path
        try:
            config = utils.config_from_yaml_resource(config_path)
        except FileNotFoundError:
            config = utils.config_from_yaml_file(config_path)

        paramaters_block = config['parameters']
        comp_block = config['components'][comp_name]

        # get pixelization and dtype
        healpix = paramaters_block['healpix']
        try:
            dtype = paramaters_block['dtype']
        except KeyError:
            dtype = np.float32

        value, resource_path, polstr=None, mult_fact=1, 

        scalar_verbose_str = f'Fixing component {comp_name} (model {sed_name}) param {param_name} to {float(value)}'
        fullpath_verbose_str = f'Fixing component {comp_name} (model {sed_name}) param {param_name} to data at {value}'
        resource_verbose_str = f'Fixing component {comp_name} (model {sed_name}) param {param_name} to {value} template'
        value = utils.parse_maplike_value(
            value, healpix, resource_path, dtype=dtype, verbose=verbose
            )

    @property
    def model(self):
        return self._nm_dict['inv_cov_mat']

    @model.setter
    def model(self, value):
        # anything that can broadcast is OK, so do icov[:]= instead of icov=
        self._nm_dict['inv_cov_mat'][:] = value

