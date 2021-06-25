# a set of standard functions that take an arbitrarily-shaped object (with kwargs) and 
# returns a result that will broadcast with a default shape

import numpy as np

from tacos import utils
from pixell import enmap

def simple_upgrade(arr, factor=1):
    # enmap upgrade needs a wcs
    arr = enmap.enmap(arr, copy=False) # copying is slow!
    arr = enmap.upgrade(arr, factor)
    return np.asarray(arr)

def P_to_QU(arr, axis=-3):
    arr = utils.atleast_nd(arr, 3)
    if arr.shape[axis] == 1:
        return np.repeat(arr, 2, axis) 
    elif arr.shape[axis] == 2:
        # assume the 1st element is I
        return np.repeat(arr, [1, 2], axis=axis)
    else:
        raise ValueError(f'Axis {axis} length must be 1 or 2; is {arr.shape[axis]}')

def simple_downgrade(arr, factor=1, op=np.mean):
    # enmap downgrade needs a wcs
    arr = enmap.enmap(arr, copy=False) # copying is slow!
    arr = enmap.downgrade(arr, factor, op=op)
    return np.asarray(arr)