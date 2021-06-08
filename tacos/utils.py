# helper utility functions
from os import O_EXCL
from pixell import enmap, utils

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

def symmetrize(arr, axis1=0, axis2=1, method='average'):
    """Symmetrizes the input array along the specified axes.

    Parameters
    ----------
    arr : array
        Array to be symmetrized
    axis1 : int, optional
        The first axis e.g. the rows, by default 0
    axis2 : int, optional
        The second axis e.g. the columns, by default 1
    method : str, optional
        Symmetrization method, either 'average' or 'from_triu',
        by default 'average'

    Returns
    -------
    array
        A symmetrized version of the input

    Notes
    -----
    If the method is 'average,' the returned array is (a + a.T)/2, which
    is the simple average of the off diagonals.

    If the method is 'from_triu,' the returned array is (a + a.T - diag(a)),
    which first sets the lower triangle to 0, and then replaces it with
    the upper triangle.
    """
    # store wcs if imap is ndmap
    if hasattr(arr, 'wcs'):
        is_enmap = True
        wcs = arr.wcs
    else:
        is_enmap = False

    # sensibility check
    assert axis2 > axis1, 'axis2 must be greater than axis1'

    # get size of axis to be symmetrized
    assert arr.shape[axis1] == arr.shape[axis2], 'Must symmetrize about axes of equal dimension'
    N = arr.shape[axis1]

    if method == 'average':
        # take simple average
        arr = (arr + np.moveaxis(arr, (axis1, axis2), (axis2, axis1)))/2

    elif method == 'from_triu':
        # diagonal goes to last axis, see np.diagonal
        diagonal = np.diagonal(arr, axis1=axis1, axis2=axis2)

        # this puts the diagonal into something of shape (N, N, ...)
        # so it can broadcast against arr
        diagonal = np.einsum('ab,...b->ab...', np.eye(N, dtype=int), diagonal)

        # take the triu of the array
        # first move axes to front of array
        arr = np.moveaxis(arr, (axis1, axis2), (0, 1))
        tril_idxs = np.tril_indices(N, -1)
        arr[tril_idxs] *= 0

        # symmetrize the array: add transpose then subtract diagonal
        arr = (arr + np.moveaxis(arr, (0, 1), (1, 0))) - diagonal

        # move axes back
        arr = np.moveaxis(arr, (0, 1), (axis1, axis2))

    if is_enmap:
        arr =  enmap.ndmap(arr, wcs)
    
    return arr    

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

def get_coadd_map_icovar(imap, icovar, imap_split_axis=-4, imap_icovar_axis=-3, icovar_split_axis=-5, icovar_axis1=-4, icovar_axis2=-3,
                        return_icovar_coadd=False):
    """Returns the icovar-weighted coadd of the imaps. The coaddition
    occurs along the specified 'split' axes.

    Parameters
    ----------
    imap : array-like
        Maps to coadd
    icovar : array-like
        Inverse covariance at each pixel
    imap_split_axis : int, optional
        The axis in imaps corresponding to the independent splits, by default -4
    imap_icovar_axis : int, optional
        The axis in imaps corresponding to the covaried component, by default -3
    icovar_split_axis : int, optional
        The axis in icovar corresponding to the independent splits,, by default -5
    icovar_axis1 : int, optional
        The axis in imaps corresponding to the first covaried component, by default -4
    icovar_axis2 : int, optional
        The axis in imaps corresponding to the second covaried component, by default -3
    return_icovar_coadd : bool, optional
        Whether to return the icovar of the coadd, by default False

    Returns
    -------
    array-like or tuple of array-like
        The coadded map, with at least 3 dimensions. If return_icovar_coadd is True,
        then also the inverse covariance of the coadd map, with at least 4 dimensions.
    """

    # store wcs if imap is ndmap
    if hasattr(imap, 'wcs'):
        is_enmap = True
        wcs = imap.wcs
    else:
        is_enmap = False

    # first move the axes into a standard position
    ndim = imap.ndim
    imap = atleast_nd(imap, 4)
    imap = np.moveaxis(imap, (imap_split_axis, imap_icovar_axis), (-4, -3))

    icovar = atleast_nd(icovar, 5)
    icovar = np.moveaxis(icovar, (icovar_split_axis, icovar_axis1, icovar_axis2), (-5, -4, -3))

    # build matrix products
    # icovar shape is (splits, pol1, pol2, ...), imap shape is (splits, pol1, ...)
    num = np.einsum('...iabxy,...ibxy->...axy', icovar, imap)
    den = np.sum(icovar, axis=-5)
    iden = eigpow(den, -1, axes=(-4, -3))

    # perform final dot product and return
    omap = np.einsum('...abxy,...bxy->...axy', iden, num)

    if is_enmap:
        omap = enmap.ndmap(omap, wcs)
    if return_icovar_coadd:
        omap = (omap, den)

    return omap

def eigpow(A, e, axes=[-2, -1], rlim=None, alim=None):
    """A hack around pixell.utils.eigpow which upgrades the data
    precision to at least double precision if necessary prior to
    operation.
    """
    # store wcs if imap is ndmap
    if hasattr(A, 'wcs'):
        is_enmap = True
        wcs = A.wcs
    else:
        is_enmap = False

    dtype = A.dtype
    
    # cast to double precision if necessary
    if np.dtype(dtype).itemsize < 8:
        A = np.asanyarray(A, dtype=np.float64)
        recast = True
    else:
        recast = False

    O = utils.eigpow(A, e, axes=axes, rlim=rlim, alim=alim)

    # cast back to input precision if necessary
    if recast:
        O = np.asanyarray(O, dtype=dtype)

    if is_enmap:
        O =  enmap.ndmap(O, wcs)

    return O

def lmax_from_wcs(wcs, method='zach'):
    """Get lmax from wcs, either "k-space" or "CAR" lmax (by method "zach" or "adri" respectively).
    """
    if method == 'zach':
        num = 180
    elif method == 'adri':
        num = 90
    else:
        raise ValueError(f'Only "zach" or "adri" methods supported')
    den = abs(wcs.wcs.cdelt[1])
    return int(num/den)

def fwhm_from_ell_bell(ell, bell):
    assert ell > 0, "ell must be greather than 0"
    assert bell < 1, "bell must be less than 1"
    sigma = np.sqrt(-2 * np.log(bell) / (ell*(ell+1)))
    fwhm = sigma * np.sqrt(8 * np.log(2))
    return fwhm

