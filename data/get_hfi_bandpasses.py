import numpy as np
import argparse
import os

from astropy.io import fits
from astropy.table import Table
import h5py

from tacos import constants, units

opj = os.path.join

parser = argparse.ArgumentParser(
    'Convert HFI bandpass file from PLA to standardized hdf5 file.')
parser.add_argument('filepath', type=str, help='Path to HFI_RIMO_R3.00.fits file')
parser.add_argument('odir', type=str, help='Path to output dir')
args = parser.parse_args()

bands = ['100', '143', '217', '353', '545', '857', '353_psb']
bandpasses = {}

with fits.open(args.filepath) as hdul:

    for bidx, band in enumerate(bands):

        # Skip first three HDUs.
        data = Table(hdul[bidx+3].data)

        # Go from [cm]^-1 to Hz.
        nu = data['WAVENUMBER'] * 1e2 * constants.clight()
        bandpass = data['TRANSMISSION']

        bandpasses[band] = {}
        bandpasses[band]['nu'] = nu
        bandpasses[band]['bandpass'] = bandpass

oname = 'bandpass_hfi_all_all_all.hdf5'
with h5py.File(opj(args.odir, oname), 'w') as hfile:

    for band in bands:

        hgroup = hfile.create_group(band)
        hgroup.create_dataset('nu', data=bandpasses[band]['nu'])
        hgroup.create_dataset('bandpass', data=bandpasses[band]['bandpass'])
        

