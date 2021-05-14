'''
Functions for unit conversions.
'''
import numpy as np

from tacos import constants as cs
    
def convert_rj_to_cmb(bandpass, nu):
    '''
    Return scalar unit conversion factor going from K_RJ to K_CMB taking
    bandpass integration into account.

    Parameters
    ----------
    bandpass : (nfreq) array
        Bandpass 
    nu : (nfreq) array
        Monotonically increasing array of frequencies in Hz.

    Returns
    -------
    unit_conv : scalar
        Unit conversion from K_RJ (brightness or RJ temp) to K_CMB 
        (thermodynamic temperature).
    '''
    
    nu = np.atleast_1d(nu)
    bandpass = np.atleast_1d(bandpass)

    # Note, typo in beyondplanck 2011.05609: eq. 39 uses h instead of kb.
    # We use Eq. 46 in Jarosik 2003 astro-ph/0301164.

    numerator = np.trapz(bandpass, x=nu)
    w_prime = dw_dt(nu)
    denominator = np.trapz(w_prime * bandpass, x=nu)

    return numerator / denominator

def integrate_over_bandpass(signal, bandpass, nu, axis=-1):
    '''
    Integrate signal over bandpass.

    Paramters
    ---------
    signal : (..., nfreq) or (nfreq) array
        Signal as function if frequency
    bandpass : (nfreq) array
        Bandpass
    nu : (nfreq) array
        Monotonically increasing array of frequencies in Hz.
    axis : int, optional
        Frequency axis in signal array.

    Returns
    -------
    int_signal : (...) array or int
    '''

    # Reshape bandpass to allow broadcasting.
    bc_shape = np.ones(signal.ndim, dtype=int)
    bc_shape[axis] = bandpass.size
    bandpass = bandpass.reshape(tuple(bc_shape))

    return np.trapz(signal * bandpass, x=nu, axis=axis)

def dw_dt(nu, temp=None):
    '''
    Return derivative of w function (defined in Eq. 38 in Jarosik 2003 
    astro-ph/0301164) with respect to a given temperature. 

    w = h nu  / (e^x - 1),    x = h nu / kB / T.
    
    dw / dT = x^2 e^x / (e^x - 1)^2.

    Arguments
    ---------
    nu : (nfreq) array or float
        Monotonically increasing array of frequencies in Hz.
    cmb_temp : float, optional
        Evaluate derivate at this temperature in Kelvin, defaults
        to CMB temperature.

    Returns
    -------
    dw_dt : (nfreq) array or float
        Derivative of w respect to temperature, evaluated at input 
        temperature (dimensionless).
    '''

    if temp is None:
        temp = cs.cmb_temp()
        
    nu = np.asarray(nu)
    ndim_in = nu.ndim
    if ndim_in == 0:
        nu = np.atleast_1d(nu)
    dtype_in = nu.dtype

    hplanck = cs.hplanck()
    kboltz = cs.kboltz()
    clight = cs.clight()

    xx = hplanck * nu / (kboltz * temp)
    xx = xx.astype(np.float64)

    # Small limit corresponds to approx. 60 Hz.
    mask_small = xx < 1e-10
    # Large limit corresponds to approx. 1130 GHz.
    mask_large = xx > 200

    expx = np.zeros_like(xx)
    np.exp(xx, out=expx, where=~(mask_small | mask_large))

    out = np.ones_like(xx)
    np.divide(xx ** 2 * expx, (expx - 1) ** 2, out=out)

    # Works because lim_x->0 x^2 e^x / (e^x - 1)^2 = 1.
    out[mask_small] = 1

    # Works because lim_nu->inf nu^2 x^2 e^x / (e^x - 1)^2 = 0.
    out[mask_large] = 0

    if ndim_in == 0:
        return out[0]
    else:
        return out
    
def db_dt(nu, temp=None):
    '''
    Return derivative of blackbody function with respect to temperature,
    evaluated at a given temperature.

    Arguments
    ---------
    nu : (nfreq) array or float
        Monotonically increasing array of frequencies in Hz.
    cmb_temp : float, optional
        Evaluate derivate at this temperature in Kelvin, defaults
        to CMB temperature.

    Returns
    -------
    db_dt : (nfreq) array or float
        Derivative of blackbody function with respect to temperature,                                              
        evaluated at input temperature. In units of W / (sr m^2 Hz K).
    '''

    kboltz = cs.kboltz()
    clight = cs.clight()
    out = dw_dt(nu, temp=temp)

    out *= 2 * kboltz * nu ** 2 / clight ** 2

    return out
