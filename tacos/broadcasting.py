# a set of standard functions that take an arbitrarily-shaped object (with kwargs) and 
# returns a result that will broadcast with a default shape

import numpy as np

from tacos import utils
from pixell import enmap
import healpy as hp 

REGISTERED_FUNCS = {}

@utils.register(REGISTERED_FUNCS)
def P_to_QU(arr, axis=0, healpix=False):
    if healpix:
        arr = utils.atleast_nd(arr, 2, axis=axis)
    else:
        arr = utils.atleast_nd(arr, 3, axis=axis)
    if arr.shape[axis] == 1:
        return np.repeat(arr, 2, axis) 
    elif arr.shape[axis] == 2:
        # assume the 1st element is I
        return np.repeat(arr, [1, 2], axis=axis)
    else:
        raise ValueError(f'Axis {axis} length must be 1 or 2; is {arr.shape[axis]}')

@utils.register(REGISTERED_FUNCS)
def simple_upgrade(arr, factor=1, healpix=False):
    if healpix:
        nside_in = hp.npix2nside(arr.shape[-1])
        nside_out = factor * nside_in
        arr = hp.ud_grade(arr, nside_out)
    else:
        # enmap upgrade needs a wcs
        arr = enmap.enmap(arr, copy=False) # copying is slow!
        arr = enmap.upgrade(arr, factor)
    return np.asarray(arr)

@utils.register(REGISTERED_FUNCS)
def simple_downgrade(arr, factor=1, op=np.mean, healpix=False):
    if healpix:
        nside_in = hp.npix2nside(arr.shape[-1])
        nside_out = nside_in // factor
        arr = hp.ud_grade(arr, nside_out)
    else:
        # enmap downgrade needs a wcs
        arr = enmap.enmap(arr, copy=False) # copying is slow!
        arr = enmap.downgrade(arr, factor, op=op)
        return np.asarray(arr)