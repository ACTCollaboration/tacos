import numpy as np
from scipy import interpolate as interp
import os

from pixell import enmap

from tacos import data, utils, models, config

module_config = utils.config_from_yaml_resource('configs/mixing_matrix.yaml')

method_order_key = {
    'linear': 1,
    'quadratic': 2,
    'cubic': 3
} 

class Element:

    def __init__(self, channel, component):
        
        self.channel = channel
        self.component = component
        
        method = module_config['interpolation']['method']
        order = method_order_key[method]

        # get span of each param(s) in the component
        model_name = component.model.__class__.__name__

        spans = {}
        for param in component.params:
            comp_block = module_config['interpolation'][model_name][param]
            low = comp_block['low']
            high = comp_block['high']
            N = comp_block['N']
            spans[param] = np.linspace(low, high, N) # follows params order

        # build interpolator
        nu = channel.bandpass.nu

        # model has no non-linear parameters
        if len(spans) == 0:
            signal = component.model(nu)
            y = channel.bandpass.integrate_signal(signal) # this is one number!
            def interpolator(*args, **kwargs):
                return y
            interpolator_call_kwargs = {}

        # model has one non-linear parameter
        elif len(spans) == 1:
            signal = component.model(nu, **spans)
            y = channel.bandpass.integrate_signal(signal) # this will have shape len(spans.values()[0])
            interpolator = interp.interp1d(*spans.values(), y, kind=method, bounds_error=True)
            interpolator_call_kwargs = {}

        # model has two non-linear parameter
        elif len(spans) == 2:
            meshed_spans = np.meshgrid(*spans.values(), indexing='ij', sparse=True)
            meshed_spans = {k: v for k, v in zip(spans.keys(), meshed_spans)}
            signal = component.model(nu, **meshed_spans) 
            y = channel.bandpass.integrate_signal(signal) # shape is (len(spans.values()[0]), len(spans.values()[1]))
            interpolator = interp.RectBivariateSpline(*spans.values(), y, kx=order, ky=order)
            interpolator_call_kwargs = {'grid': False}

        else:
            raise NotImplementedError('Only up to 2-parameter models implemented so far')

        self.interpolator = interpolator
        self.interpolator_call_kwargs = interpolator_call_kwargs

    def __call__(self, **kwargs):

        # we need to build a list of parameter values in the proper order
        param_values = []
        for param in self.component.params:
        
            # broadcast active params, or grab already-broadcasted fixed params
            if param in self.component.active_params:
                param_values.append(self.component.param_broadcasters[param](kwargs[param]))
            else:
                assert param not in kwargs, f'Param {param} is fixed but was passed as a kwarg'
                param_values.append(self.component.fixed_params[param])

        # interpolate, broadcast with component broadcaster
        res = self.interpolator(*param_values, **self.interpolator_call_kwargs)
        return self.channel.bandpass.rj_to_cmb * self.component.comp_broadcaster(res)

class MixingMatrix:
    
    def __init__(self, channels, components, shape, wcs=None, dtype=np.float32):

        nchan = len(channels)
        ncomp = len(components)

        self.shape = shape
        utils.check_shape(self.shape)

        self._wcs = wcs # if this is None, return array as-is (ie, healpix), see matrix property
        self._dtype = dtype

        self._elements = {}
        for comp in components:
            self._elements[comp.name] = []
            for chan in channels:
                self._elements[comp.name].append(Element(chan, comp))
        
        self._matrix = np.zeros((nchan, ncomp) + shape, dtype=dtype)

    def __call__(self, chain=None, iteration=-1, **comp_params):
        if chain is not None:
            assert chain.shape == self._matrix.shape[2:], \
                f'Params object shape {chain.shape} must equal matrix shape {self.matrix.shape[2:]}'
            assert len(comp_params) == 0, \
                'If Chain instance is passed, cannot also pass implicit component parameters'
            _, _, comp_params = chain.get_samples(sel=np.s_[iteration])

        # update Elements by component
        for compidx, comp_name in enumerate(self._elements):
            active_params = comp_params.get(comp_name, {})
            for chanidx, element in enumerate(self._elements[comp_name]):
                self._matrix[chanidx, compidx] = element(**active_params)

        return self.matrix

    @classmethod
    def load_from_config(cls, config_path, verbose=True):
        _, channels, components, _, shape, wcs, kwargs = _load_all_from_config(config_path, verbose=verbose)
        return cls(channels, components, shape, wcs, **kwargs)

    @property
    def matrix(self):
        if self._wcs is None:
            return self._matrix
        else:
            return enmap.ndmap(self._matrix, self._wcs)

def _load_all_from_config(config_path, load_channels=True, load_components=True, verbose=True):
    try:
        config = utils.config_from_yaml_resource(config_path)
    except FileNotFoundError:
        config = utils.config_from_yaml_file(config_path)

    # get list of channels
    channels = []
    if load_channels:
        for instr, bands in config['channels'].items():
            for band, kwargs in bands.items():
                if kwargs == 'None':
                    kwargs = {}
                channels.append(data.Channel(instr, band, **kwargs))
            
    # get list of components
    components = []
    if load_components:
        for comp_name in config['components']:
            components.append(models.Component.load_from_config(config_path, comp_name, verbose=verbose))  

    # get pol, shape, wcs, dtype
    params_block = config['parameters']
    polstr, shape, wcs, kwargs = utils.parse_parameters_block(params_block, verbose=verbose)

    # get name from config stem
    config_base, _ = os.path.splitext(config_path)
    name = os.path.basename(config_base)
    return name, channels, components, polstr, shape, wcs, kwargs

def get_exact_mixing_matrix(channels, components, shape, wcs=None, dtype=np.float32):
    '''
    Return mixing matrix for given frequency bands and signal
    components.

    Parameters
    ----------
    bandpasses : (nj) array-like of BandPass objects
        Bandpass for each frequency band.
    betas : (ncomp) array or (ncomp, ny, nx) enmap.
        Spectral indices per components.
    dtype : type
        Dtype for output.
    
    Returns
    -------
    mixing_mat : (nj, ncomp, ...) array or ndmap
        Mixing matrix.
    '''

    nchan = len(channels)
    ncomp = len(components)

    m = np.zeros((nchan, ncomp, *shape), dtype=dtype)

    for chanidx, chan in enumerate(channels):
        u_conv = chan.bandpass.rj_to_cmb
        for compidx, comp in enumerate(components):
            res = u_conv * chan.bandpass.integrate_signal(comp)
            m[chanidx, compidx] = res

    if wcs:
        m = enmap.ndmap(m, wcs)

    return m