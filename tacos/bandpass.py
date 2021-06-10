import numpy as np
from scipy.interpolate import interp1d
import os 

from pixell import enmap
import h5py

from tacos import units, utils

config = utils.config_from_yaml_resource('configs/bandpass_config.yaml')

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
                N = N
                self.nu = np.linspace(float(nu_low), float(nu_high), int(N))
            elif nu_iterator == 'arange':
                delta_nu = delta_nu
                self.nu = np.arange(float(nu_low), float(nu_high), float(delta_nu))

            # get new bandpass
            bandpass = interp1d(nu, bandpass, kind='linear', bounds_error=False, fill_value=0.0)
            bandpass = bandpass(self.nu)

        # normalize the bandpass to have an integral of 1 over frequency
        bandpass = bandpass / np.trapz(bandpass, x=self.nu)

        # Compute 1D interpolation and store as attribute.
        self.bandpass = interp1d(self.nu, bandpass, kind='linear', bounds_error=False, fill_value=0.0)

        # Store unit conversion as an attribute
        self.c_fact_rj_to_cmb = units.convert_rj_to_cmb(
            self.bandpass, self.nu)

    def integrate_signal(self, signal, axis=-1):
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

        # Convert callables to arrays, if necessary
        bandpass = self.bandpass(self.nu)
        
        if callable(signal):
            signal = signal(self.nu)
        signal = np.atleast_1d(signal)

        # Reshape bandpass to allow broadcasting.
        bc_shape = np.ones(signal.ndim, dtype=int)
        bc_shape[axis] = bandpass.size
        bandpass = bandpass.reshape(tuple(bc_shape))

        return np.trapz(signal * bandpass, x=self.nu, axis=axis)

    @classmethod
    def load_act_bandpass(cls, filename, band, array=None):
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

        bandpass_kwargs = config[band]
        return cls(bandpass, nu, **bandpass_kwargs)

    @classmethod
    def load_planck_bandpass(cls, filename, band, psb_only=True,
                          nu_sq_corr=True):
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
        hfi_bandpass : bandpass.BandPass instance
            Bandpass instance for requested HFI band.

        Raises
        ------
        ValueError
            If band is not recognized.        
        '''

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

        bandpass_kwargs = config[bandname]
        return cls(bandpass, nu, **bandpass_kwargs)

    @classmethod
    def load_wmap_bandpass(cls, filename, band):
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
                
        bandpass_kwargs = config[band]
        return cls(bandpass, nu, **bandpass_kwargs)

def get_mixing_matrix(bandpasses, betas, dtype=np.float32):
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
    mixing_mat : (nj, ncomp) array or (nj, ncomp, ny, nx) enmap
        Mixing matrix.
    '''

    raise NotImplementedError

    nj = len(bandpasses)
    ncomp = betas.shape[0]

    if isinstance(betas, enmap.ndmap):
        mixing_mat = enmap.zeros((nj,) + betas.shape, wcs=betas.wcs)
    else:
        mixing_mat = np.zeros((nj, ncomp), dtype=dtype)

    for jidx in range(nj):

        u_conv = bandpasses[jidx].conv_fact_rj_to_cmb()
        bandpass = bandpasses[jidx].bandpass
        nu = bandpass.nu

        # NOTE, betas need nu dependence, right?
        mixing_mat[jidx] = units.integrate_over_bandpass(
            betas, bandpass, nu)

    return mixing_mat
