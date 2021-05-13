# helper utility functions

import numpy as np

def atleast_nd(arr, n, axis=None):
    """Return a buffer whose new shape has at least n dimensions.

    Parameters
    ----------
    arr : array-like
        Input array
    n : int
        Minimum number of dimensions in returned array
    axis : int or iterable, optional
        Axis or axes where new dimensions will located in returned array, by default None.
        If None, axes are prepended until the number of dimensions is n.

    Returns
    -------
    array-like
        Reshaped array
    """
    arr = np.asanyarray(arr)
    if (axis is None) or (arr.ndim >= n):
        oaxis=tuple(range(n - arr.ndim)) # prepend the dims or do nothing if n <= arr.ndim
    else:
        axis = np.atleast_1d(axis)
        assert (n - arr.ndim) >= len(axis), 'More axes than dimensions to add'
        oaxis = tuple(range(n - arr.ndim - len(axis))) + tuple(axis) # prepend the extra dims
    return np.expand_dims(arr, oaxis)

def get_coadd_map(imap, ivar, axis=-4):
    """Returns the ivar-weighted coadd of the the imaps. The coaddition
    occurs along the specified axis.

    Parameters
    ----------
    imap : array-like
        Maps to coadd
    ivar : array-like
        Coadd weights
    axis : int, optional
        Axis of coaddition, by default -4

    Returns
    -------
    array-like
        Coadded map, with at least four dimensions.
    """
    imap = atleast_nd(imap, 4) # make 4d by prepending
    ivar = atleast_nd(ivar, 4)

    # due to floating point precision, the coadd is not exactly the same
    # as a split where that split is the only non-zero ivar in that pixel
    coadd = np.nan_to_num( 
        np.sum(imap * ivar, axis=axis, keepdims=True) / np.sum(ivar, axis=axis, keepdims=True) # splits along -4 axis
        )

    # find pixels where exactly one split has a nonzero ivar
    single_nonzero_ivar_mask = np.broadcast_to(np.sum(ivar!=0, axis=axis) == 1, coadd.shape)
    
    # set the coadd in those pixels to be equal to the imap value of that split (ie, avoid floating
    # point errors in naive coadd calculation)
    coadd[single_nonzero_ivar_mask] = np.sum(imap * (ivar!=0), axis=axis, keepdims=True)[single_nonzero_ivar_mask]
    return coadd