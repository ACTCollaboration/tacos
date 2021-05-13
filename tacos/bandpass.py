import numpy as np

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
        '''

        with h5py.File(filename + '.hdf5', 'r') as f:
            factors = f['factors'][()]
            rule = f['rule'][()]
            weights = f['weights'][()]
            ells = f['ells_full'][()]
            name = f['name'][()].decode("utf-8") 

        return cls(factors, rule, weights, ells, name)

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
