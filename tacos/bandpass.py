import numpy as np
import os 

from pixell import enmap

from compsep import units

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
    '''
    def __init__(self, bandpass, nu):
        
        self.bandpass = bandpass / np.trapz(bandpass, x=nu)
        self.nu = nu

        # Can compute conv factor at init, right?
        # Can also compute interpolation table at init.

    def conv_fact_rj_to_cmb(self):

        return units.convert_rj_to_cmb(self.bandpass, self.nu)

    def compute_interpolation(self):
        
        # Given range of betas, compute interpolation lookup table.
        # For single beta: 1d table, for two betas 2d table etc.
        # But are you doing this for all sky components? Or one
        # component at a time? Perhaps not here then.
        
        raise NotImplementedError()

    def integrate_over_bandpass(self):
        pass

    @classmethod
    def load_hfi_bandpass(cls, filename, band, psb_only=True,
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
                bandanme = band

            bandpass = hfile[f'{bandname}/bandpass'][()]
            nu = hfile[f'{bandname}/nu'][()]

            if nu_sq_corr:
                bandpass *= nu ** 2

        return cls(bandpass, nu)

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
            for didx, da in enumerate(range(1 + nda)):
                for rad in range(1, 3):
                    
                    bandname = band + str(da) + str(rad)

                    if didx == 0:
                        bandpass = hfile[f'{bandname}/bandpass'][()]
                        nu = hfile[f'{bandname}/nu'][()]
                    else:
                        bandpass += hfile[f'{bandname}/bandpass'][()]
            bandpass /= nda * 2
                
        return cls(bandpass, nu)

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

    nj = len(bandpasses)
    ncomp = betas.shape[0]

    if isinstance(betas, enmap.ndmap):
        mixing_mat = enmap.zeros((nj,) + betas.shape), wcs=betas.wcs)
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
