import numpy as np
from scipy.interpolate import interp1d
import os 

from pixell import enmap
import h5py

# class is overkill, just need 3 load functions that return array

def load_planck_beam(filename, band, psb_only=True):

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

        return bell
