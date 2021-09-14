import numpy as np
from scipy.interpolate import interp1d
import os  

import h5py

from tacos import units, utils

config = utils.config_from_yaml_resource('configs/bandpass.yaml')
tophat_config = utils.config_from_yaml_resource('configs/bandpass_pysm_tophat.yaml')

class BandPass():
    '''
    Bandpass instance contains normalized bandpass and conversion
    factors.

    Parameters
    ----------
    bandpass : (nfreq) array
        Bandpass.
    nu : (nfreq) array
        Frequencies in Hz.
    trim_zeros : bool, optional
        If True, remove trailing and leading zeros from nu (and corresponding locations 
        from bandpass) before other operations, by default False.
    rtol : float, optional
        See notes, default is 0
    atol : float, optional
        See notes, default is 0
    nu_iterator : str, optional
        Either "linspace" or "arange"
    nu_low : float, optional
        Lower limit of integration, default is nu.min()
    nu_high : float, optional
        Upper limit of integration, default is nu.max()
    N : int, optional
        If nu_iterator is "linspace," the number of steps, default is 500.
    delta_nu: float, optional
        If nu_iterator is "arange," the size of each step, default is 100MHz

    Attributes
    ----------
    bandpass : (nfreq) array
        Bandpass normalized such that integral over frequencies gives 1.
    nu : (nfreq) array
        Frequencies in Hz.
    c_fact_rj_to_cmb : float
        Conversion factor to go from a flat RJ spectrum to thermodynamic
        units.    

    Notes
    -----
    If trim_zeros is True, the definition of "zero" is less than or equal to
    rtol*bandpass.max() + atol

    If rtol or atol are greater than 0, or if a nu_iterator is provided, such that
    the bandpass changes from the raw data in any way, the output bandpass is renormalized
    such that its integral is still 1 over its domain.
    '''

    def __init__(self, bandpass, nu, trim_zeros=False, rtol=0., atol=0., nu_iterator=None, nu_low=None, nu_high=None, N=500, delta_nu=0.1e9):
        # Because kwargs may be supplied via a yaml file, numeric kwargs are explicitly cast to correct type

        # Trim leading and trailing zeros
        if trim_zeros:
            nu, bandpass = utils.trim_zeros(nu, ref=bandpass, rtol=float(rtol), atol=float(atol), return_ref=True)
        
        if nu_iterator is None:
            self.nu = nu
        else:
            # get bounds of new nu
            if nu_low is None:
                nu_low = nu.min()
            if nu_high is None:
                nu_high = nu.max()

            # get new nu
            if nu_iterator == 'linspace':
                self.nu = np.linspace(float(nu_low), float(nu_high), int(N))
            elif nu_iterator == 'arange':
                self.nu = np.arange(float(nu_low), float(nu_high), float(delta_nu))

            # get new bandpass by interpolating against old nu
            bandpass = interp1d(nu, bandpass, kind='linear', bounds_error=False, fill_value=0.0)
            bandpass = bandpass(self.nu)

        # normalize the bandpass to have an integral of 1 over frequency
        bandpass = bandpass / np.trapz(bandpass, x=self.nu)

        # Compute 1D interpolation and store as attribute.
        self.bandpass = interp1d(self.nu, bandpass, kind='linear', bounds_error=False, fill_value=0.0)

        # Store unit conversion as an attribute
        self.rj_to_cmb = units.convert_rj_to_cmb(self.bandpass, self.nu)
        self.cmb_to_rj = units.convert_cmb_to_rj(self.bandpass, self.nu)

    def integrate_signal(self, signal, signal_kwargs=None, axis=-1):
        '''
        Integrate signal over bandpass.

        Parameters
        ---------
        signal : (..., nfreq) or (nfreq) array or callable
            Signal as function of frequency
        axis : int, optional
            Frequency axis in signal array.

        Returns
        -------
        int_signal : (...) array or int
            Integrated signal.
        '''

        # Convert callables to arrays
        bandpass = self.bandpass(self.nu)
        
        if callable(signal):
            if signal_kwargs:
                signal = signal(self.nu, **signal_kwargs)
            else:
                signal = signal(self.nu)
        signal = np.atleast_1d(signal)

        # Reshape bandpass to allow broadcasting.
        bc_shape = np.ones(signal.ndim, dtype=int)
        bc_shape[axis] = bandpass.size
        bandpass = bandpass.reshape(tuple(bc_shape))
        return np.trapz(signal * bandpass, x=self.nu, axis=axis)

    @classmethod
    def load_act_bandpass(cls, filename, band, array=None, **kwargs):
        '''
        Read ACT bandpass file and return class instance.

        Parameters
        ----------
        filename : str
            Absolute path to file.
        band : str
            Band name, either "f090", "f150", or "f220"
        array : str, optional
            ACT array, e.g. 'pa2_f150', by default None

        Returns
        -------
        act_bandpass : bandpass.BandPass instance
            Bandpass instance for requested ACT array.

        Raises
        ------
        ValueError
            If array is not recognized.        
        '''
        # assert the instrument and filename match
        head, tail = os.path.split(filename)
        assert head.split('/')[-1] == 'act'
        assert tail.split('_')[1] == 'act'

        bands = ['f090', 'f150', 'f220']
        arrays = ['pa4_f150', 'pa4_f220', 'pa5_f090', 'pa5_f150', 'pa6_f090', 'pa6_f150']
        
        if band not in bands:
            raise ValueError(f'Requested band : {band} not recognized. '
                             f'Pick from {bands}.')

        if array is not None and array not in arrays:
            raise ValueError(f'Requested array : {array} not recognized. '
                             f'Pick from {arrays}.')
        
        if not os.path.splitext(filename)[1]:
            filename = filename + '.hdf5'

        with h5py.File(filename, 'r') as hfile:
            
            # Use average of bandpasses for now.
            band_arrays = []
            for ar in arrays:
                if band in ar:
                    band_arrays.append(ar)

            for i, ar in enumerate(band_arrays):
                if i == 0:
                    bandpass = hfile[f'{ar}/bandpass'][()]
                    nu = hfile[f'{ar}/nu'][()]
                else:
                    bandpass += hfile[f'{ar}/bandpass'][()]
            bandpass /= len(band_arrays)

        bandpass_kwargs = config['act'][band]
        return cls(bandpass, nu, **bandpass_kwargs)

    @classmethod
    def load_planck_bandpass(cls, filename, band, psb_only=True,
                          nu_sq_corr=True, **kwargs):
        '''
        Read Planck HFI bandpass file and return class instance.

        Parameters
        ----------
        filename : str
            Absolute path to file.
        band : str
            Planck HFI band, e.g. '143'.
        psb_only : bool, optional
            Load special bandpasses for polarization sensitive bolometers
            if available.
        nu_sq_corr : bool, optional
            Preprocess bandpasses by applying nu ** 2 correction factor.

        Returns
        -------
        planck_bandpass : bandpass.BandPass instance
            Bandpass instance for requested HFI band.

        Raises
        ------
        ValueError
            If band is not recognized.        
        '''
        # assert the instrument and filename match
        head, tail = os.path.split(filename)
        assert head.split('/')[-1] == 'planck'
        assert tail.split('_')[1] == 'planck' 

        bands = ['100', '143', '217', '353', '545', '857']
        if band not in bands:
            raise ValueError(f'Requested band : {band} not recognized. '
                             f'Pick from {bands}.')
        
        if not os.path.splitext(filename)[1]:
            filename = filename + '.hdf5'

        with h5py.File(filename, 'r') as hfile:
            
            if psb_only and band == '353':
                bandname = band + '_psb'
            else:
                bandname = band

            bandpass = hfile[f'{bandname}/bandpass'][()]
            nu = hfile[f'{bandname}/nu'][()]

            if nu_sq_corr:
                bandpass *= nu ** 2

        bandpass_kwargs = config['planck'][bandname]
        return cls(bandpass, nu, **bandpass_kwargs)

    @classmethod
    def load_wmap_bandpass(cls, filename, band, **kwargs):
        '''
        Read WMAP bandpass file and return class instance.

        Parameters
        ----------
        filename : str
            Absolute path to file.
        band : str
            WMAP band, e.g. 'Q'.

        Returns
        -------
        wmap_bandpass : bandpass.BandPass instance
            Bandpass instance for requested WMAP band.

        Raises
        ------
        ValueError
            If band is not recognized.        
        '''
        # assert the instrument and filename match
        head, tail = os.path.split(filename)
        assert head.split('/')[-1] == 'wmap'
        assert tail.split('_')[1] == 'wmap'

        bands = ['K', 'Ka', 'Q', 'V', 'W']
        if band not in bands:
            raise ValueError(f'Requested band : {band} not recognized. '
                             f'Pick from {bands}.')
        
        if not os.path.splitext(filename)[1]:
            filename = filename + '.hdf5'

        with h5py.File(filename, 'r') as hfile:
            
            if band in ['K', 'Ka']:
                nda = 1
            elif band in ['Q', 'V']:
                nda = 2
            else:
                nda = 4

            # Use average of bandpasses for now.
            for didx, da in enumerate(range(1, 1 + nda)):
                for rad in range(1, 3):
                    
                    bandname = band + str(da) + str(rad)

                    if didx == 0 and rad == 1:
                        bandpass = hfile[f'{bandname}/bandpass'][()]
                        nu = hfile[f'{bandname}/nu'][()]
                    else:
                        bandpass += hfile[f'{bandname}/bandpass'][()]
            bandpass /= nda * 2
                
        bandpass_kwargs = config['wmap'][band]
        return cls(bandpass, nu, **bandpass_kwargs)

    @classmethod
    def load_pysm_bandpass(cls, filename, instr, band, nu_sq_corr=True, tophat=False, **bandpass_kwargs):
        """Read bandpass file corresponding to provided instrument and generate a pysm-tophat
        band over the same frequency range.

        Parameters
        ----------
        filename : str
            Absolute path to file.
        instr : str
            One of "act", "planck", or "wmap". Must match filename.
        band : str
            Band belonging to one of the instruments.
        nu_sq_corr : bool, optional
            Preprocess bandpasses by applying nu ** 2 correction factor, by default True
        tophat : bool, optional
            Load bandpass_kwargs for the pysm band from bandpass_pysm_tophat.yaml instead
            of bandpass.yaml, by default False

        Returns
        -------
        pysm_bandpass : bandpass.BandPass instance
            Bandpass instance of requested instrument, band; possibly a tophat of the same.
        """
        # assert the instrument and filename match
        head, tail = os.path.split(filename) 
        assert head.split('/')[-1] == instr
        assert tail.split('_')[1] == instr

        if instr == 'act':
            bandpass_obj = cls.load_act_bandpass(filename, band, **bandpass_kwargs)
        elif instr == 'planck':
            bandpass_obj = cls.load_planck_bandpass(filename, band, nu_sq_corr=True, **bandpass_kwargs)
        elif instr == 'wmap':
            bandpass_obj = cls.load_wmap_bandpass(filename, band, **bandpass_kwargs)

        nu = bandpass_obj.nu
        bandpass = bandpass_obj.bandpass(nu)

        # need to make another dummy bandpass_obj with the pysm config before 
        # final modifications like a tophat to properly define the domain
        if tophat:
            bandpass_kwargs = tophat_config['pysm'][band] 
        else:
            bandpass_kwargs = config['pysm'][band]
        
        bandpass_obj = cls(bandpass, nu, **bandpass_kwargs)
        nu = bandpass_obj.nu
        bandpass = bandpass_obj.bandpass(nu)

        # get final bandpass
        if tophat:
            bandpass = np.ones_like(nu)
        if nu_sq_corr:
            bandpass *= nu ** 2

        return cls(bandpass, nu, **bandpass_kwargs)