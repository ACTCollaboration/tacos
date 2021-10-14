import os

from tacos import utils, data, component
   

class Config:

    # this is like a glorified struct object which just packages parsed
    # config items together for other constructors to cherry-pick from
    
    def __init__(self, config_path, load_channels=True, load_components=True, verbose=True):
        config_dict = utils.config_from_yaml(config_path)

        # get name from config stem
        config_base, _ = os.path.splitext(config_path)
        self._name = os.path.basename(config_base)

        # get pol, shape, wcs, dtype, ...
        global_config_block = utils.GlobalConfigBlock(config_path, verbose=verbose)

        self._polstr = global_config_block.polstr 
        self._healpix = global_config_block.healpix
        self._shape = global_config_block.shape
        self._wcs = global_config_block.wcs 
        self._dtype = global_config_block.dtype 
        self._num_steps = global_config_block.num_steps
        self._max_N = global_config_block.max_N
        self._linsampler = global_config_block.linsampler   

        # get list of channels
        self._channels = []
        if load_channels:
            for instr, bands in config_dict['channels'].items():
                for band, channel_kwargs in bands.items():
                    if (channel_kwargs is None) or (channel_kwargs == 'None'):
                        channel_kwargs = {}
                    self._channels.append(data.Channel(
                        instr, band, polstr=self._polstr, healpix=self._healpix, **channel_kwargs
                        ))
                
        # get list of components and possible their priors
        self._components = []
        if load_components:
            for comp_name in config_dict['components']:
                self._components.append(
                    component.Component.load_from_config(config_path, comp_name, verbose=verbose)
                    )  

    @property
    def name(self):
        return self._name

    @property
    def channels(self):
        return self._channels

    @property
    def components(self):
        return self._components

    @property
    def polstr(self):
        return self._polstr

    @property
    def healpix(self):
        return self._healpix

    @property
    def shape(self):
        return self._shape

    @property
    def wcs(self):
        return self._wcs

    @property
    def dtype(self):
        return self._dtype

    @property
    def num_steps(self):
        return self._num_steps
    
    @property
    def max_N(self):
        return self._max_N

    @property
    def linsampler(self):
        return self._linsampler