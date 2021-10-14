import numpy as np

from tacos import chain as _chain, mixing_matrix as _mixing_matrix, config as _config
from tacos.sampling import linear, nonlinear

class GibbsSampler:

    # Holds metadata about the sampling run, as well as the necessary objects: a Chain, 
    # MixingMatrix, LinSampler, and NonLinSampler (possibly)

    def __init__(self, chain, mixing_matrix, linsampler=None, nonlinsampler=None,
                init_amplitudes=None, init_params=None, num_steps=None, dtype=None):
        self._chain = chain
        self._mixing_matrix = mixing_matrix
        self._linsampler = linsampler
        self._nonlinsampler = nonlinsampler
        
        self._num_steps = num_steps if num_steps else 1000
        self._dtype = dtype if dtype else np.float32

        self._data = None
        self._noise_models = None

        num_chan, num_comp, num_pol = self._mixing_matrix.shape[:3]
        self._num_chan = num_chan
        self._num_comp = num_comp
        self._num_pol = num_pol

        self._prev_amplitudes_sample = init_amplitudes
        self._prev_params_sample = init_params

    def step(self):
        
        # somehow we need a starting point, if beta, and whether or not priors.
        # do we always start with a beta sample, if available? I think so
        params_sample = self._chain.get_empty_params_sample()
        if self._nonlinsampler is not None:
            for comp, param in self._chain.paramsiter():
                params_sample[comp][param] = self._nonlinsampler(self._prev_amplitudes_sample)

        # get the mixing matrix for this call. this is fast because
        # only updated params are actually evaluated
        M = self._mixing_matrix(**params_sample)

        amplitudes_sample = self._linsampler(M)

        self._chain.add_samples(weights=[1,1], amplitudes=amplitudes_sample, params=params_sample)

        self._prev_amplitudes_sample = amplitudes_sample
        self._prev_params_sample = params_sample

    def run(self, step_per_save=None):
        if step_per_save is None:
            step_per_save = np.inf
        else:
            assert step_per_save > 0, 'Step_per_save must be positive'
            step_per_save = int(step_per_save)
        
        step_num = 0
        for i in range(self._num_steps):
            self.step()
            step_num += 1
            if step_num == step_per_save:
                self._chain.write_samples()
                step_num = 0

    # def log_like(self, M, a):
    #     n = np.einsum('jca...,ca...->ja...', M, a)
    #     n = self.data - n
    #     Ninvn = np.array(
    #         [self._noise_models[i].filter(self._data[i]) for i in range(self._num_chan)],
    #         dtype=self._dtype
    #         )
    #     chi_d = 

    @classmethod
    def load_from_config(cls, config_path, verbose=True):
        chain = _chain.Chain.load_from_config(config_path, verbose=False)
        mixing_matrix = _mixing_matrix.MixingMatrix.load_from_config(config_path, verbose=verbose)
        config_obj = _config.Config(config_path, load_channels=False, load_components=False, verbose=False)
        channels = mixing_matrix.channels
        components = mixing_matrix.components
        num_steps = config_obj.num_steps
        dtype = config_obj.dtype
        linsampler_class_name = config_obj.linsampler
        linsampler_class = linear.REGISTERED_SAMPLERS[linsampler_class_name]
        
        # construct the linsampler instance
        noise_models = [channel.noise_model for channel in channels]
        data = [channel.map for channel in channels]
        prior_models = [component.comp_prior_model for component in components]
        prior_means = [component.comp_prior_mean for component in components]
        linsampler = linsampler_class(
            mixing_matrix, noise_models, data, prior_models=prior_models,
            prior_means=prior_means, dtype=dtype
            )
        return cls(
            chain=chain, mixing_matrix=mixing_matrix, linsampler=linsampler,
            num_steps=num_steps, dtype=dtype
            )