import numpy as np
from scipy.interpolate import interp1d
import os 

from pixell import enmap
import h5py

# class is overkill, just need 3 load functions that return array

def load_act_beam(filename, band, array=None):
    """Read ACT beam file and return b(ell) array. If beam file
    specifies non-exhaustive ells, return cubic spline interpolation over
    np.arange(lmax + 1)

    Parameters
    ----------
    filename : str
        Absolute path to file.
    band : str
        ACT band, e.g. 'f090'.
    array : str, optional
        ACT array, e.g. 'pa2_f150', by default None

    Returns
    -------
    array
        The beam window function evaluated at every ell.

    Raises
    ------
    ValueError
        If band is not recognized. 
    """

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
                ell = hfile[f'{ar}/ell'][()]
                bell = hfile[f'{ar}/bell'][()]
            else:
                bell += hfile[f'{ar}/bell'][()]
        bell /= len(band_arrays)
                
        # interpolate up to max(ell)
        lmax = ell[-1]
        bell_interp = interp1d(ell, bell, kind='cubic')
        bell = bell_interp(np.arange(lmax+1)) # up to and including lmax

        return bell / np.max(bell)

def load_planck_beam(filename, band, psb_only=True):
    """Read Planck HFI beam file and return b(ell) array. If beam file
    specifies non-exhaustive ells, return cubic spline interpolation over
    np.arange(lmax + 1)

    Parameters
    ----------
    filename : str
        Absolute path to file.
    band : str
        Planck HFI band, e.g. '143'.
    psb_only : bool, optional
        Load special beams for polarization sensitive bolometers
        if available, by default True

    Returns
    -------
    array
        The beam window function evaluated at every ell.

    Raises
    ------
    ValueError
        If band is not recognized. 
    """

    bands = ['100', '143', '217', '353']
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

        ell = hfile[f'{bandname}/ell'][()]
        bell = hfile[f'{bandname}/bell'][()]

        # interpolate up to max(ell)
        lmax = ell[-1]
        bell_interp = interp1d(ell, bell, kind='cubic')
        bell = bell_interp(np.arange(lmax+1)) # up to and including lmax

        return bell / np.max(bell)

def load_wmap_beam(filename, band):
    """Read WMAP beam file and return b(ell) array. If beam file
    specifies non-exhaustive ells, return cubic spline interpolation over
    np.arange(lmax + 1)

    Parameters
    ----------
    filename : str
        Absolute path to file.
    band : str
        WMAP band, e.g. 'Q'.

    Returns
    -------
    array
        The beam window function evaluated at every ell.

    Raises
    ------
    ValueError
        If band is not recognized. 
    """

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

        # Use average of beams for now.
        for da in range(1, nda + 1):
            bandname = band + str(da)
            if da == 1:
                ell = hfile[f'{bandname}/ell'][()]
                bell = hfile[f'{bandname}/bell'][()]
            else:
                bell += hfile[f'{bandname}/bell'][()]
        bell /= nda
                
        # interpolate up to max(ell)
        lmax = ell[-1]
        bell_interp = interp1d(ell, bell, kind='cubic')
        bell = bell_interp(np.arange(lmax+1)) # up to and including lmax

        return bell / np.max(bell)

