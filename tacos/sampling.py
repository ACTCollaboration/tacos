import numpy as np
from scipy.interpolate import interp1d
import os 

from pixell import enmap
import h5py

from tacos import units, utils

def get_mixing_matrix(channels, components, dtype=np.float32):
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

    if hasattr(channels[0].map, 'wcs'):
        is_enmap = True
        wcs = channels[0].map.wcs
    else:
        is_enmap = False

    m_shape = (nchan, ncomp) + channels[0].map.shape
    m = np.zeros(m_shape, dtype=dtype)

    for chanidx, chan in enumerate(channels):
        u_conv = chan.bandpass.rj_to_cmb
        for compidx, comp in enumerate(components):
            res = u_conv * chan.bandpass.integrate_signal(comp)
            m[chanidx, compidx] = res

    if is_enmap:
        m = enmap.ndmap(m, wcs)

    return m