# helper utility functions
from pixell import enmap, curvedsky
from enlib import array_ops

import healpy as hp 
import camb
import numpy as np
import yaml

import hashlib
from concurrent import futures
import multiprocessing
import pkgutil
from ast import literal_eval
import os

### tacos ###

def get_cpu_count():
    """Number of available threads, either from environment variable 'OMP_NUM_THREADS' or physical CPU cores"""
    try:
        nthread = int(os.environ['OMP_NUM_THREADS'])
    except (KeyError, ValueError):
        nthread = multiprocessing.cpu_count()
    return nthread

def concurrent_standard_normal(size=1, nchunks=100, nthread=0, seed=None, dtype=np.float32, complex=False):
    """Draw standard normal (real or complex) random variates concurrently.
    
    Parameters
    ----------
    size : int or iterable, optional
        The shape to draw random numbers into, by default 1.
    nchunks : int, optional
        The number of concurrent subdraws to make, by default 100. 
        These draws are concatenated in the output; therefore, the
        output changes both with the seed and with nchunks.
    nthread : int, optional
        Number of concurrent threads, by default 0. If 0, the result
        of get_cpu_count().
    seed : int or iterable-of-ints, optional
        Random seed to pass to np.random.SeedSequence, by default None.
    dtype : np.dtype, optional
        Data type of output if real, or of each real and complex,
        component, by default np.float32. Must be a 4- or 8-byte
        type.
    complex : bool, optional
        If True, return a complex random variate, by default False.
    
    Returns
    -------
    ndarray
        Real or complex standard normal random variates in shape 'size'
        with each real and/or complex part having dtype 'dtype'. 
    
    Raises
    ------
    ValueError
        If the dtype does not have 4 or 8 bytes.
    """
    # get size per chunk draw
    totalsize = np.prod(size, dtype=int)
    chunksize = np.ceil(totalsize/nchunks).astype(int)

    # get seeds
    ss = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in ss.spawn(nchunks)]
    
    # define working objects
    out = np.empty((nchunks, chunksize), dtype=dtype)
    if complex:
        out_imag = np.empty_like(out)

    # perform multithreaded execution
    if nthread == 0:
        nthread = get_cpu_count()
    executor = futures.ThreadPoolExecutor(max_workers=nthread)

    def _fill(arr, start, stop, rng):
        rng.standard_normal(out=arr[start:stop], dtype=dtype)
    
    fs = [executor.submit(_fill, out, i, i+1, rngs[i]) for i in range(nchunks)]
    futures.wait(fs)

    if complex:
        fs = [executor.submit(_fill, out_imag, i, i+1, rngs[i]) for i in range(nchunks)]
        futures.wait(fs)

        # if not concurrent, casting to complex takes 80% of the time for a complex draw
        if np.dtype(dtype).itemsize == 4:
            idtype = np.complex64
        elif np.dtype(dtype).itemsize == 8:
            idtype = np.complex128
        else:
            raise ValueError('Input dtype must have 4 or 8 bytes')
        imag_vec = np.full((nchunks, 1), 1j, dtype=idtype)
        out_imag = concurrent_op(np.multiply, out_imag, imag_vec, nchunks=nchunks, nthread=nthread)
        out = concurrent_op(np.add, out, out_imag, nchunks=nchunks, nthread=nthread)

    # return
    out = out.reshape(-1)[:totalsize]
    return out.reshape(size)

def concurrent_op(op, a, b, *args, chunk_axis_a=0, chunk_axis_b=0, nchunks=100, nthread=0, **kwargs):
    """Perform a numpy operation on two arrays concurrently.
    
    Parameters
    ----------
    op : numpy function
        A numpy function to be performed, e.g. np.add or np.multiply
    a : ndarray
        The first array in the operation.
    b : ndarray
        The second array in the operation
    chunk_axis_a : int, optional
        The axis in a over which the operation may be applied
        concurrently, by default 0.
    chunk_axis_b : int, optional
        The axis in b over which the operation may be applied
        concurrently, by default 0.
    nchunks : int, optional
        The number of chunks to loop over concurrently, by default 100.
    nthread : int, optional
        The number of threads, by default 0.
        If 0, use output of get_cpu_count().
    Returns
    -------
    ndarray
        The result of op(a, b, *args, **kwargs), except with the axis
        corresponding to the a, b chunk axes located at axis-0.
    
    Notes
    -----
    The chunk axes are what a user might expect to naively 'loop over'. For
    maximum efficiency, they should be long. They must be of equal size in
    a and b.
    """
    # move axes to standard positions
    a = np.moveaxis(a, chunk_axis_a, 0)
    b = np.moveaxis(b, chunk_axis_b, 0)
    assert a.shape[0] == b.shape[0], f'Size of chunk axis must be equal, got {a.shape[0]} and {b.shape[0]}'
    
    # get size per chunk draw
    totalsize = a.shape[0]
    chunksize = np.ceil(totalsize/nchunks).astype(int)

    # define working objects
    # in order to get output shape, dtype, must get shape, dtype of op(a[0], b[0])
    out_test = op(a[0], b[0], *args, **kwargs)
    out = np.empty((totalsize, *out_test.shape), dtype=out_test.dtype)

    # perform multithreaded execution
    if nthread == 0:
        nthread = get_cpu_count()
    executor = futures.ThreadPoolExecutor(max_workers=nthread)

    def _fill(start, stop):
        op(a[start:stop], b[start:stop], *args, out=out[start:stop], **kwargs)
    
    fs = [executor.submit(_fill, i*chunksize, (i+1)*chunksize) for i in range(nchunks)]
    futures.wait(fs)

    # return
    return out

def hash_str(str, ndigits=9):
    """Turn a qid string into an ndigit hash, using hashlib.sha256 hashing"""
    return int(hashlib.sha256(str.encode('utf-8')).hexdigest(), 16) % 10**ndigits

### I/O ###

# copied from soapack.interfaces
def config_from_yaml_file(filename):
    """Returns a dictionary from a yaml file given by absolute filename.
    """
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config

def config_from_yaml_resource(resource):
    """Returns a dictionary from a yaml file given by the resource name (relative to tacos package).
    """
    f = pkgutil.get_data('tacos', resource).decode()
    config = yaml.safe_load(f)
    return config

def config_from_yaml(filename_or_resource):
    try:
        config = config_from_yaml_resource(filename_or_resource)
    except FileNotFoundError:
        config = config_from_yaml_file(filename_or_resource)
    return config

data_paths = config_from_yaml_file(os.environ['HOME'] + '/.soapack.yml')['tacos']
extensions = config_from_yaml_resource('configs/data.yaml')['extensions']

def data_fn_str(product=None, instr=None, band=None, id=None, set=None, notes=None):
    """Returns a generic data filename, of format '{product}_{instr}_{band}_{id}_{set}{notes}.{ext}'"""
    if not notes:
        notes = ''
    else:
        notes = '_' + notes
    data_fn_str_template = '{product}_{instr}_{band}_{id}_{set}{notes}.{ext}'
    return data_fn_str_template.format(
        product=product, instr=instr, band=band, id=id, set=set, notes=notes, ext=extensions[product]
        )

def data_dir_str(product, instr=''):
    """Returns a generic data directory, of format '{product_dir}{instr}/'"""
    data_dir_str_template = '{product_dir}{instr}/'
    
    # there are multiple kinds of covmat products
    if product in ['icovar']:
        product = 'covmat'
    product_dir = data_paths[f'{product}_path']
    
    return data_dir_str_template.format(
        product_dir=product_dir, instr=instr
    )

def data_fullpath_str(product=None, instr=None, band=None, id=None, set=None, notes=None):
    path = data_dir_str(product, instr)
    path += data_fn_str(product, instr, band, id, set, notes)
    return path

def polstr2polidxs(polstr):
    if polstr:
        polidxs = np.array(['IQU'.index(char) for char in polstr])
    else:
        polidxs = np.arange(3)
    return polidxs

def read_geometry_from(s, healpix=False):
    """Return geometry from a file s. Shape trimmed to return only map axes.
    """
    if healpix:
        shape = hp.read_map(s).shape
        return (shape[-1],) # just want the map shape, no pol
    else:
        shape, wcs = enmap.read_map_geometry(s)
        return shape[-2:], wcs # just want the map shape, no pol

def parse_maplike_value(value, healpix, resource_path, dtype=None, 
                        scalar_verbose_str=None, fullpath_verbose_str=None, resource_verbose_str=None,
                        verbose=False):
    dtype = dtype if dtype else np.float32
    
    if isinstance(value, (int, float)):
        if verbose:
            print(scalar_verbose_str)
        value = float(value)

    # if string, first see if it exists as a fullpath to a file. if so, load it directly
    elif isinstance(value, str):
        if os.path.exists(value):
            if verbose:
                print(fullpath_verbose_str)
            if healpix:
                value = hp.read_map(value, field=None, dtype=dtype)
            else:
                value = enmap.read_map(value)
        
        # if not fullpath, try loading it as a preexisting resource      
        else:
            if verbose:
                print(resource_verbose_str)
            if healpix:
                value += '_healpix'

            resource_fullpath = resource_path + value + '.' + extensions['resource']
            if healpix:
                value = hp.read_map(resource_fullpath, field=None, dtype=dtype)
            else:
                value = enmap.read_map(resource_fullpath)
    
    return value

class GlobalConfigBlock:

    def __init__(self, config_path, verbose=True):
        """Parse the global block of a config file and return the polarization string,
        map shape (including polarization components), and optionally the wcs and other
        kwargs to pass to MixingMatrix(...), Params(...) constructors.

        Parameters
        ----------
        global_block : dict
            A dictionary corresponding to the parameters block of a configuration file. See
            notes for required entries.
        verbose : bool
            Print informative statements.

        Notes
        -----
        Some keys in params_block are required. These are:
            
            healpix : bool
            pol : a string in {IQU | IQ | IU | QU | I | Q | U}

        Some keys in params_block are members of a set, only one of which must be provided:

            {geometry_from : path | shape : tuple, wcs_from: path | nside: int}

        Some keys in params_block are optional. These are:

            dtype : string formatter for numpy datatype
            max_N : integer number of max samples in chain
        """
        config_dict = config_from_yaml(config_path)
        global_block = config_dict['global']

        polstr = global_block['pol']
        assert polstr in ['IQU', 'IQ', 'IU', 'QU', 'I', 'Q', 'U']
        self._polstr = polstr

        # get whether to assume maps are in healpix
        healpix = global_block['healpix']
        if verbose:
            pixstr = 'HEALPix' if healpix else 'CAR'
            print(f'Maps assumed to be in {pixstr} pixelization')
        self._healpix = healpix

        # if geometry_from is provided, grab that info
        if 'geometry_from' in global_block:
            assert 'shape' not in global_block and 'wcs_from' not in global_block and 'nside' not in global_block, \
                'Path to geometry, and shape, path to wcs, or nside provided; ambiguous'
            if verbose:
                print(f'Reading map geometry from path {global_block["geometry_from"]}')
            geometry = read_geometry_from(global_block['geometry_from'], healpix)

            # get the geometry-like objects for car vs. healpix
            try:
                shape, wcs = geometry # car
            except ValueError:
                shape, wcs = geometry, None # healpix

        # otherwise, grab shape and possibly wcs info
        elif 'shape' in global_block:
            assert not healpix, 'Maps in HEALPix; cannot supply shape or wcs in config'
            assert 'geometry_from' not in global_block, \
                'Path to geometry, and shape, path to wcs_provided; ambiguous'
            if verbose:
                print(f'Reading shape from config directly, found {global_block["shape"]}')
            shape = literal_eval(global_block['shape'])

            if verbose:
                print(f'Reading wcs from path {global_block["wcs_from"]}')
            _, wcs = read_geometry_from(global_block['wcs_from'], healpix=False)

        # otherwise, grab nside
        elif 'nside' in global_block:
            assert healpix, 'Maps in HEALPix; must supply nside if not path to geometry'
            assert 'geometry_from' not in global_block, \
                'Path to geometry, and nside; ambiguous'
            if verbose:
                print(f'Reading nside from config directly, found {global_block["nside"]}')
            shape = tuple([hp.nside2npix(global_block['nside'])])
            wcs = None

        # otherwise, throw error
        else:
            raise ValueError('Must supply sufficient geometry information in the config')
        shape = (len(polstr),) + shape
        self._shape = shape
        self._wcs = wcs

        # optional attributes
        self._dtype = global_block.get('dtype')
        self._num_steps = global_block.get('num_steps')
        self._max_N = global_block.get('max_N')

    @property
    def polstr(self):
        return self._polstr

    @property
    def healpix(self):
        return self._healpix

    @property
    def shape(self):
        return self._shape

    @property
    def wcs(self):
        return self._wcs

    @property
    def dtype(self):
        return self._dtype

    @property
    def num_steps(self):
        return self._num_steps
    
    @property
    def max_N(self):
        return self._max_N

def eplot(x, *args, fname=None, show=False, **kwargs):
    """Return a list of enplot plots. Optionally, save and display them.

    Parameters
    ----------
    x : ndmap
        Items to plot
    fname : str or path-like, optional
        Full path to save the plots, by default None. If None, plots are
        not saved.
    show : bool, optional
        Whether to display plots, by default False
    **kwargs : dict
        Optional arguments to pass to enplot.plot

    Returns
    -------
    list
        A list of enplot plot objects.
    """
    from pixell import enplot
    plots = enplot.plot(x, **kwargs)
    if fname is not None:
        enplot.write(fname, plots)
    if show:
        enplot.show(plots)
    return plots

def eshow(x, *args, fname=None, return_plots=False, **kwargs):
    """Show enplot plots of ndmaps. Optionally, save and return them.

    Parameters
    ----------
    x : ndmap
        Items to plot
    fname : str or path-like, optional
        Full path to save the plots, by default None. If None, plots are
        not saved.
    return_plots : bool, optional
        Whether to return plots, by default False

    Returns
    -------
    list or None
        A list of enplot plot objects, only if return_plots is True.
    """
    res = eplot(x, *args, fname=fname, show=True, **kwargs)
    if return_plots:
        return res

### Arrays Ops ###

def trim_zeros(ar1, ref=None, rtol=0., atol=0., return_ref=False):
    """Remove elements from ar1 based on indices corresponding to leading and trailing
    zeros in ref. 

    Parameters
    ----------
    ar1 : array
        Array to trim
    ref : array, optional
        Indices of leading and trailing zeros in ref are removed from ar1, by default ar1.
    rtol : float, optional
    atol : float, optional
    return_ref: bool, optional
        Return the trimmed reference, by default False.

    Returns
    -------
    array
        Trimmed version of ar1

    Notes
    -----
    The definition of "zero" is less than or equal to rtol*ref.max() + atol, by default 0.
    """
    if ref is None:
        ref = ar1
    ar1 = np.atleast_1d(ar1)
    ref = np.atleast_1d(ref)
    assert ar1.ndim == 1 and ref.ndim == 1, 'Currently only supports single-axis operation'

    # get cut value based on rtol and atol
    cut = rtol*ref.max() + atol

    # apply cut
    nonzero = np.nonzero(ref > cut)[0]
    start, stop = nonzero[0], nonzero[-1]

    if return_ref:
        out = (ar1[start:stop+1], ref[start:stop+1])
    else:
        out = ar1[start:stop+1]
    return out

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
        oaxis = tuple(range(n - arr.ndim)) # prepend the dims or do nothing if n <= arr.ndim
    else:
        axis = np.atleast_1d(axis)
        assert (n - arr.ndim) >= len(axis), 'More axes than dimensions to add'
        oaxis = tuple(range(n - arr.ndim - len(axis))) + tuple(axis) # prepend the extra dims
    return np.expand_dims(arr, oaxis)

def expand_all_arg_dims(*args):
    """Append an extra dimension to all args, even if scalar."""
    return (np.atleast_1d(a)[..., None] for a in args)

def expand_all_kwarg_dims(**kwargs):
    """Append an extra dimension to all values in kwargs, even if scalar"""
    return {k: np.atleast_1d(v)[..., None] for k, v in kwargs.items()}

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
    is the simple average of the off-diagonals.

    If the method is 'from_triu,' the returned array is (a + a.T - diag(a)),
    where a has its lower triangle first set to 0, and then replaced with
    its upper triangle.
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

def eigpow(A, e, axes=[-2, -1]):
    """A hack around enlib.array_ops.eigpow which upgrades the data
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

    array_ops.eigpow(A, e, axes=axes, copy=False)

    # cast back to input precision if necessary
    if recast:
        A = np.asanyarray(A, dtype=dtype)

    if is_enmap:
        A = enmap.ndmap(A, wcs)

    return A

### Maps ###

def check_shape(shape):
    assert len(shape) == 3 or len(shape) == 2 # car and healpix
    assert shape[0] in (1,2,3), 'Polarization must have 1, 2, or 3 components' 

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

    # sensibility check
    assert icovar_axis2 > icovar_axis1, 'axis2 must be greater than axis1'

    # first move the axes into a standard position
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

def get_icovar_noise_sim(icovar, icovar_axis1=-4, icovar_axis2=-3, seed=None, mult_fact=1):
    """Given a pixel-space inverse covariance matrix, draw a noise simulation
    directly from it.

    Parameters
    ----------
    icovar : ndmap or array
        The inverse-covariance matrix to draw from
    icovar_axis1 : int, optional
        The axis in imaps corresponding to the first covaried component, by default -4
    icovar_axis2 : int, optional
        The axis in imaps corresponding to the second covaried component, by default -3
    seed : int or tuple of ints, optional
        The seed of the random draw, by default None
    mult_fact: int, optional
        Multiply icovar to manually adjust level of inverse-covariance, useful for
        testing, by default 1

    Returns
    -------
    ndmap or array
        A simulation drawn from the covariance matrix corresponding to icovar.

    Notes
    -----
    If icovar has shape (*s1,N1,*s2,N2,...,ny,nx) where N1 and N2 are the covaried
    components (must be equal) and ny, nx are the map dimensions, the sample will
    have shape (*(s1+s2),N2,...,ny,nx).

    The covariance matrix must be the inverse of icovar, by definition. This function
    uses eigpow internally, which performs an SVD on icovar, with eigenvalues raised 
    to the -0.5 power, applied to standard-normal draws in the right shape.
    """
    # store wcs if imap is ndmap
    if hasattr(icovar, 'wcs'):
        is_enmap = True
        wcs = icovar.wcs
    else:
        is_enmap = False

    # sensibility check
    assert icovar_axis2 > icovar_axis1, 'axis2 must be greater than axis1'

    np.random.seed(seed)
    
    # move axes into a standard position, then
    # shape of sample will be same as icovar, except "removing" the "first icovar axis" i.e. -4
    icovar = np.moveaxis(icovar, (icovar_axis1, icovar_axis2), (-4, -3))
    oshape = icovar.shape[:-4] + icovar.shape[-3:]

    # we want covar**0.5, which is icovar**-0.5, then
    # draw a map-space sample into the right shape
    # finally, move axes back if necessary
    icovar = eigpow(mult_fact * icovar, -0.5, axes=(icovar_axis1, icovar_axis2))
    x = np.random.randn(*oshape).astype(icovar.dtype)
    res = np.einsum('...ijyx,...jyx->...iyx', icovar, x)
    res = np.moveaxis(res, -3, icovar_axis2)

    if is_enmap:
        res = enmap.ndmap(res, wcs)
    return res

def get_cmb_sim(shape, wcs, H0=67.9, lmax=6_000, dtype=np.float32, seed=None):
    """Get a realization of the CMB, using CAMB default parameters up to the 
    specified H0 and lmax. Units are in uK_CMB. 

    Parameters
    ----------
    shape : tuple
        Output shape of map. Will be promoted to length-3 if length-2, and truncated
        to length-3 if longer than length-3.
    wcs : wcs
    H0 : float, optional
    lmax : int, optional
        Bandlimit of realization, by default 6_000
    dtype : dtype, optional
        Data type of map, by default np.float32
    seed : int or tuple of int, optional
        Seed for realization, by default None

    Returns
    -------
    ndmap
        Map of the realization, of shape (ncomp, ny,nx)
    """
    shape = atleast_nd(np.empty(shape), 3).shape[-3:]

    # get power spectra
    params = camb.model.CAMBparams()
    params.set_cosmology(H0=H0)
    params.set_for_lmax(lmax)
    res = camb.get_results(params)
    spectra = res.get_cmb_power_spectra(lmax=lmax, spectra=('total',), CMB_unit='muK', raw_cl=True)['total']

    # modify the shape of spectra (lmax+1,4) to (3,3,lmax+1)
    TT, EE, BB, TE = spectra.T
    spectra = np.array([
        [TT, TE, np.zeros_like(TT)],
        [TE, EE, np.zeros_like(TT)],
        [np.zeros_like(TT), np.zeros_like(TT), BB]
    ])

    # get rand map
    return curvedsky.rand_map(shape, wcs, spectra, lmax=lmax, dtype=dtype, seed=seed)

### Harmonic/Fourier ###

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
    """Given a point bell(ell) for some beam window function, return the fwhm corresponding to hp.gauss_beam.
    """
    assert ell > 0, "ell must be greather than 0"
    assert bell < 1, "bell must be less than 1"
    sigma = np.sqrt(-2 * np.log(bell) / (ell*(ell+1)))
    fwhm = sigma * np.sqrt(8 * np.log(2))
    return fwhm