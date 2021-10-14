import numpy as np
from scipy import interpolate as interp

from pixell import enmap
from enlib import bench

from tacos import utils, config

module_config = utils.config_from_yaml_resource('configs/mixing_matrix.yaml')

method_order_key = {
    'linear': 1,
    'quadratic': 2,
    'cubic': 3
} 

class _Element:

    def __init__(self, channel, component):
        
        self.channel = channel
        self.component = component
        
        method = module_config['interpolation']['method']
        order = method_order_key[method]

        # get span of each param(s) in the component
        sed_name = component.sed.__class__.__name__

        spans = {}
        for param in component.params:
            comp_block = module_config['interpolation'][sed_name][param]
            low = comp_block['low']
            high = comp_block['high']
            N = comp_block['N']
            spans[param] = np.linspace(low, high, N) # follows params order

        # build interpolator
        nu = channel.bandpass.nu

        # sed has no non-linear parameters
        if len(spans) == 0:
            signal = component.sed(nu)
            y = channel.bandpass.integrate_signal(signal) # this is one number!
            def interpolator(*args, **kwargs):
                return y
            interpolator_call_kwargs = {}

        # sed has one non-linear parameter
        elif len(spans) == 1:
            signal = component.sed(nu, **spans)
            y = channel.bandpass.integrate_signal(signal) # this will have shape len(spans.values()[0])
            interpolator = interp.interp1d(*spans.values(), y, kind=method, bounds_error=True)
            interpolator_call_kwargs = {}

        # sed has two non-linear parameter
        elif len(spans) == 2:
            meshed_spans = np.meshgrid(*spans.values(), indexing='ij', sparse=True)
            meshed_spans = {k: v for k, v in zip(spans.keys(), meshed_spans)}
            signal = component.sed(nu, **meshed_spans) 
            y = channel.bandpass.integrate_signal(signal) # shape is (len(spans.values()[0]), len(spans.values()[1]))
            interpolator = interp.RectBivariateSpline(*spans.values(), y, kx=order, ky=order)
            interpolator_call_kwargs = {'grid': False}

        else:
            raise NotImplementedError('Only up to 2-parameter seds implemented so far')

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
    
    def __init__(self, channels, components, shape, wcs=None, dtype=None, **comp_params):

        # shape is map shape, ie (npol, ny, nx) or (npol, npix) or (npol, nalm)

        self._channels = channels
        self._components = components
        self._comp_names  = [comp.name for comp in components]
        num_chan = len(channels)
        num_comp = len(components)

        self._element_shape = shape
        utils.check_shape(shape)

        self._wcs = wcs # if this is None, return array as-is (ie, healpix), see matrix property
        self._dtype = dtype if dtype else np.float32

        self._elements = {}
        for comp in components:
            self._elements[comp.name] = []
            for chan in channels:
                self._elements[comp.name].append(_Element(chan, comp)) # does all interpolation!
        
        self._shape = (num_chan, num_comp) + shape
        self._matrix = np.empty(self._shape, dtype=dtype)
        if self._wcs is not None:
            self._matrix = enmap.ndmap(self._matrix, self._wcs)
        
        # when initialized, need to give it something so it can build the first time
        self._init_call(**comp_params)

    def _init_call(self, **comp_params):
        # update Elements by component. do this for every element
        # the first time, so that we initialize every element
        for compidx, comp_name in enumerate(self._elements):
            comp_active_params = comp_params.get(comp_name, {})
            for chanidx, element in enumerate(self._elements[comp_name]):
                # this line is where all the broadcasting must come together!
                self._matrix[chanidx, compidx] = element(**comp_active_params) 

    def __call__(self, **comp_params):
        # update Elements by component. only want to update elements with
        # comp params that have been passed, so that successive calls
        # are as fast as possible
        for comp_name in comp_params:
            compidx = self._comp_names.index(comp_name)
            comp_active_params = comp_params[comp_name]
            for chanidx, element in enumerate(self._elements[comp_name]):
                # this line is where all the broadcasting must come together!
                self._matrix[chanidx, compidx] = element(**comp_active_params) 

        return self._matrix

    @classmethod
    def load_from_config(cls, config_path, verbose=True):
        config_obj = config.Config(config_path, verbose=verbose)
        channels = config_obj.channels
        components = config_obj.components
        shape = config_obj.shape
        wcs = config_obj.wcs
        dtype = config_obj.dtype
        return cls(channels, components, shape, wcs=wcs, dtype=dtype)

    @property
    def channels(self):
        return self._channels

    @property
    def components(self):
        return self._components

    @property
    def element_shape(self):
        return self._element_shape

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def matrix(self):
        return self._matrix

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

    if wcs is not None:
        m = enmap.ndmap(m, wcs)

    return m