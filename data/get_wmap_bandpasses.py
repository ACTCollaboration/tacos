import numpy as np
import argparse
import os

from astropy.io import fits
from astropy.table import Table
import h5py

from tacos import constants as cs, units

opj = os.path.join

parser = argparse.ArgumentParser(
    'Convert WMAP bandpass files from Lambda to standardized hdf5 file.')
parser.add_argument('idir', type=str, help='Path to input dir containing '
                    'wmap_bandpass_{band}{channel}{DA}{RM}_v5.cbp files.')
parser.add_argument('odir', type=str, help='Path to output dir')
args = parser.parse_args()

bands = ['K', 'Ka', 'Q', 'V', 'W']
bandpasses = {}

for bidx, band in enumerate(bands):

    if band in ['K', 'Ka']:
        nda = 1
    elif band in ['Q', 'V']:
        nda = 2
    else:
        nda = 4

    for da in range(1, nda + 1):
        for rad in range(1, 3):

            out = np.loadtxt(
                opj(args.idir, f'wmap_bandpass_{band}{da}{rad}_v5.cbp'),
                skiprows=10)

            out = out.T

            # Convert from GHz to Hz.
            nu = out[0] * 1e9

            # Coadd the two channels, Eq. 45 in Jarosik 2003 astro-ph/0301164.
            chn1 = out[1]
            chn2 = out[2]

            w_prime = units.dw_dt(nu)
            coadd = 0.5 * (chn1 / np.sum(chn1 * w_prime))
            coadd += 0.5 * (chn2 / np.sum(chn2 * w_prime))

            key = band + str(da) + str(rad)
            bandpasses[key] = {}
            bandpasses[key]['nu'] = nu
            bandpasses[key]['bandpass'] = coadd                        

oname = 'bandpass_wmap_all_all_all.hdf5'
with h5py.File(opj(args.odir, oname), 'w') as hfile:

    for key in bandpasses.keys():

        hgroup = hfile.create_group(key)
        hgroup.create_dataset('nu', data=bandpasses[key]['nu'])
        hgroup.create_dataset('bandpass', data=bandpasses[key]['bandpass'])
        

