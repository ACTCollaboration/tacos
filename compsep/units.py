'''
Functions for unit conversions.
'''
import numpy as np

from compsep import constants as cs

def db_dt(nu, temp=None):
    '''
    Return derivative of blackbody function with respect to temperature,
    evaluated at a given temperature.

    Arguments
    ---------
    nu : (nfreq) array or float
        Frequency in units of Hz.
    cmb_temp : float, optional
        Evaluate derivate at this temperature in Kelvin, defaults
        to CMB temperature.

    Returns
    -------
    db_dt : (nfreq) array or float
        Derivative of blackbody function with respect to temperature,                                              
        evaluated at input temperature. In units of W / (sr m^2 Hz K).
    '''

    if temp is None:
        temp = cs.cmb_temp()
        
    hplanck = cs.hplanck()
    kboltz = cs.kboltz()
    clight = cs.clight()

    xx = hplanck * nu / (kboltz * temp)

    out = 2 * hplanck * nu ** 3 / temp / clight ** 2
    out *= xx * np.exp(xx) / ((np.exp(xx) - 1.) ** 2) 

    return out
    
def convert_rj_to_cmb(bandpass, nu):
    '''
    Return scalar unit conversion factor going from K_RJ to K_CMB taking
    bandpass integration into account.

    Parameters
    ----------
    bandpass : (nfreq) array
        Bandpass 
    nu : (nfreq) array
        Frequencies corresponding to bandpass in Hz.

    Returns
    -------
    unit_conv : scalar
        Unit conversion from K_RJ (brightness or RJ temp) to K_CMB 
        (thermodynamic temperature).
    '''
    
    nu = np.atleast_1d(nu)
    bandpass = np.atleast_1d(bandpass)

    numerator = np.trapz(2 * cs.hplanck() * nu ** 2 / cs.clight * bandpass,
                         x=nu)
    denominator = np.trapz(db_dt(nu) * bandpass, x=nu)

    return numerator / denominator
    
