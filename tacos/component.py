import numpy as np

from tacos import utils, broadcasting, sky_models
from tacos.sampling import noise_models

from ast import literal_eval


template_path = utils.data_dir_str('template')


### Component Wrapper Class ###

class Component:

    def __init__(self, sed, comp_name=None, fixed_params=None, comp_broadcaster=None,
                param_broadcasters=None, param_shapes=None, comp_prior_model=None,
                comp_prior_mean=None, param_priors=None, verbose=True):
        
        self.sed = sed
        self._name = comp_name if comp_name else sed.__class__.__name__.lower()

        # store any fixed params, label all else active params
        self._params = {
            'active': [],
            'fixed': {}
        }
        if fixed_params is None:
            fixed_params = {}
        for param in sed.params:
            val = fixed_params.get(param)
            if val:
                self.fixed_params[param] = val
            else:
                self.active_params.append(param)

        # store broadcasters. these are used to guarantee a parameter of a given 
        # shape will broadcast against the mixing matrix. there can be broadcasters
        # for any parameter, which are evaluated prior to interpolation, and a 
        # broadcaster for the entire component, which is evaluated after interpolation        

        # store component-level broadcaster
        self._comp_broadcaster = lambda x: x
        if comp_broadcaster is not None:
            self._comp_broadcaster = comp_broadcaster

        # store per-parameter broadcasters, and evaluate them (once) for any fixed params
        self._param_broadcasters = {param: lambda x: x for param in sed.params}
        if param_broadcasters is None:
            param_broadcasters = {}
        for param in param_broadcasters:
            if param in self.fixed_params:
                if verbose: 
                    print(f'Broadcasting fixed param {param} and overwriting its value to ' + \
                        'broadcasted result')
                self.fixed_params[param] = param_broadcasters[param](self.fixed_params[param])
            elif param in self.active_params:
                self.param_broadcasters[param] = param_broadcasters[param]
            else:
                raise ValueError(f'Param {param} not in fixed nor active params')

        # store shape for passing onto Chain class, but it doesn't do anything here
        self._param_shapes = {}
        if param_shapes is None:
            param_shapes = {}
        for param, shape in param_shapes.items():
            if param in self.active_params:
                self.param_shapes[param] = shape

        # store the prior for the entire component (amplitudes)
        self._comp_prior_model = comp_prior_model
        self._comp_prior_mean = comp_prior_mean

        # store the priors for each active param
        self._param_priors = {}
        if param_priors is None:
            param_priors = {}
        for param, prior in param_priors.items():
            if param in self.active_params:
                self.param_priors[param] = prior

    # This function oddly has no use when things are interpolated
    # I think it will come in handy when evaluating a proposal that has gone "out of bounds"
    # TODO: implement that
    def __call__(self, nu, **kwargs):
        
        # first broadcast active params
        for param in self.active_params:
            kwargs[param] = self.param_broadcasters[param](kwargs[param])

        # then add in the already-broadcasted fixed params
        for param in self.fixed_params:
            assert param not in kwargs, f'Param {param} is fixed but was passed as a kwarg'
        kwargs.update(self.fixed_params)

        # finally, evaluate
        res = self.sed(nu, **kwargs)
        return self.comp_broadcaster(res)

    @classmethod
    def load_from_config(cls, config_path, comp_name, verbose=True):

        # can't use config.Config because that depends on sky_models.py, instead
        # use utils.GlobalConfigBlock because we only need a few global params
        global_config_block = utils.GlobalConfigBlock(config_path)
        polstr = global_config_block.polstr
        healpix = global_config_block.healpix
        shape = global_config_block.shape
        dtype = global_config_block.dtype
        
        # for non-global params, need to inspect the config in more detail
        config_dict = utils.config_from_yaml(config_path)
        comp_block = config_dict['components'][comp_name]
        
        # first get the sed of the component
        sed_name = comp_block['sed']
        try:
            nu0 = comp_block['nu0']
            sed_kwargs = {'nu0': literal_eval(nu0)} # in case format is like 23e9
        except KeyError:
            sed_kwargs = {}
        sed = sky_models.REGISTERED_SEDS[sed_name](verbose=verbose, **sed_kwargs)

        # get the component broadcasters, if any
        if 'broadcasters' in comp_block:
            comp_broadcaster = cls.parse_broadcasters(
                comp_block['broadcasters'], healpix,
                comp_name=comp_name, sed_name=sed_name, verbose=verbose
                )

        # get the component prior, if any
        if 'prior' in comp_block:
            # parse the prior model value
            prior_block = comp_block['prior']
            value = prior_block['values']['model']
            scalar_verbose_str = f'Fixing component {comp_name} prior model value to {float(value)}'
            fullpath_verbose_str = f'Fixing component {comp_name} prior model value to data at {value}'
            resource_verbose_str = f'Fixing component {comp_name} prior model value to {value} template'
            value = utils.parse_maplike_value(
                value, healpix, template_path, dtype=dtype, scalar_verbose_str=scalar_verbose_str,
                fullpath_verbose_str=fullpath_verbose_str, resource_verbose_str=resource_verbose_str,
                verbose=verbose
                )
            
            # construct the prior model instance
            prior_model_class = noise_models.REGISTERED_NOISE_MODELS[prior_block['model']]
            cov_mult_fact = prior_block.get('cov_mult_fact')
            comp_prior_model = prior_model_class(
                value, shape=shape, dtype=dtype, polstr=polstr, cov_mult_fact=cov_mult_fact
                )

            # parse the prior mean, if any
            if 'mean' in prior_block['values']:
                value = prior_block['values']['mean']
                scalar_verbose_str = f'Fixing component {comp_name} prior mean value to {float(value)}'
                fullpath_verbose_str = f'Fixing component {comp_name} prior mean value to data at {value}'
                resource_verbose_str = f'Fixing component {comp_name} prior mean value to {value} template'
                comp_prior_mean = utils.parse_maplike_value(
                    value, healpix, template_path, dtype=dtype, scalar_verbose_str=scalar_verbose_str,
                    fullpath_verbose_str=fullpath_verbose_str, resource_verbose_str=resource_verbose_str,
                    verbose=verbose
                    )
                comp_prior_mean = np.broadcast_to(comp_prior_mean, shape, subok=True)

        # get the possible fixed params, broadcasting function stack for each param, and shapes
        # of each param
        fixed_params = {}
        param_broadcasters = {}
        param_shapes = {}

        if 'params' in comp_block:
            for param_name, param_block in comp_block['params'].items():
                
                assert param_name in sed.params, f'Param {param_name} not in {sed_name} params'
                
                # if a particular param has a value, parse it
                if 'value' in param_block:
                    assert 'shape' not in param_block, 'A fixed value cannot have a config-set shape'
                    value = param_block['value']
                    scalar_verbose_str = f'Fixing component {comp_name} (model {sed_name}) param {param_name} to {float(value)}'
                    fullpath_verbose_str = f'Fixing component {comp_name} (model {sed_name}) param {param_name} to data at {value}'
                    resource_verbose_str = f'Fixing component {comp_name} (model {sed_name}) param {param_name} to {value} template'
                    fixed_params[param_name] =  utils.parse_maplike_value(
                        value, healpix, template_path, dtype=dtype, scalar_verbose_str=scalar_verbose_str,
                        fullpath_verbose_str=fullpath_verbose_str, resource_verbose_str=resource_verbose_str,
                        verbose=verbose
                        )

                # if a particular param has broadcaster(s), build the broadcasting function
                # call stack
                if 'broadcasters' in param_block:
                    param_broadcasters[param_name] = cls.parse_broadcasters(
                        param_block['broadcasters'], healpix,
                        comp_name=comp_name, sed_name=sed_name, param_name=param_name, verbose=verbose
                        )

                # if a particular param has a specific shape, load and store it
                if 'shape' in param_block:
                    assert 'value' not in param_block, 'A shaped-config component cannot have a fixed value'
                    param_shapes[param_name] = cls.parse_shape(
                        param_block['shape'],
                        comp_name=comp_name, sed_name=sed_name, param_name=param_name, verbose=verbose
                        )

        # get component
        return cls(sed, comp_name=comp_name, comp_broadcaster=comp_broadcaster,
                param_broadcasters=param_broadcasters, param_shapes=param_shapes, 
                comp_prior_model=comp_prior_model, comp_prior_mean=comp_prior_mean,
                verbose=verbose)
    
    # just helps modularize load_from_config(...)
    @classmethod
    def parse_broadcasters(cls, function_block, healpix,
        comp_name=None, sed_name=None, param_name=None, verbose=True):     
        # for each function in broadcasters, add it to the function call stack
        # with any kwargs as necessary
        func_list = []
        kwarg_list = []
        for func_name, func_kwargs in function_block.items():
            if verbose:
                msg = f'Appending {func_name} to broadcasting call stack for component {comp_name} (sed {sed_name})'
                if param_name:
                    msg += f' param {param_name}'
                print(msg)
            
            # append the function to the list
            broadcaster = getattr(broadcasting, func_name)
            func_list.append(broadcaster)

            # look for any kwargs, or append empty kwargs.
            # healpix a global parameter for the analysis, so handle it separately
            if (func_kwargs is None) or (func_kwargs == 'None'):
                kwarg_list.append({'healpix': healpix})
            else:
                func_kwargs.update({'healpix': healpix})
                kwarg_list.append(func_kwargs)

        # build a single function call stack
        def stacked_broadcaster(x):
            for i in range(len(func_list)):
                x = func_list[i](x, **kwarg_list[i])
            return x
        
        return stacked_broadcaster
    
    # just helps modularize load_from_config(...)
    @classmethod
    def parse_shape(cls, shape,
        comp_name=None, sed_name=None, param_name=None, verbose=True):        
        if verbose:
            print(f'Setting component {comp_name} (sed {sed_name}) param {param_name} sampled shape to {shape}')
        shape = literal_eval(shape) # this maps a stringified tuple to the actual tuple
        return shape

    @property
    def sed(self):
        return self._sed

    @property
    def name(self):
        return self._name

    @property
    def params(self):
        return self._sed.params

    @property
    def fixed_params(self):
        return self._params['fixed']
    
    @property
    def active_params(self):
        return self._params['active']

    @property
    def comp_broadcaster(self):
        return self._comp_broadcaster

    @property
    def param_broadcasters(self):
        return self._param_broadcasters

    @property
    def param_shapes(self):
        return self._param_shapes

    @property
    def comp_prior_model(self):
        return self._comp_prior_model

    @property
    def comp_prior_mean(self):
        return self._comp_prior_mean

    @property
    def param_priors(self):
        return self._param_priors