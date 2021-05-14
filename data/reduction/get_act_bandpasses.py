import numpy as np
import argparse
import os

from astropy.io import fits
from astropy.table import Table
import h5py

from tacos import constants, units

opj = os.path.join

parser = argparse.ArgumentParser(
    'Convert ACT bandpass file from DR5 bandpass file to standardized hdf5 file.')
parser.add_argument('filepath', type=str, 
                    help='Path to act_planck_dr5.01_s08s18_bandpasses.txt file')
parser.add_argument('odir', type=str, help='Path to output dir')
args = parser.parse_args()

arrays = ['pa1_f150', 'pa2_f150', 'pa3_f090', 'pa3_f150', 'pa4_f150',
          'pa4_f220', 'pa5_f090', 'pa5_f150', 'pa6_f090', 'pa6_f150',
          'ar1_f150', 'ar2_f220']
bandpasses = {}

bfile = np.loadtxt(args.filepath)
bfile = bfile.T

# Convert from GHz to Hz.
nu = bfile[0] * 1e9

bfile = bfile[1:]

for aidx, ar in enumerate(arrays):

    bandpass = bfile[aidx]
    bandpasses[ar] = {}
    bandpasses[ar]['nu'] = nu
    bandpasses[ar]['bandpass'] = bandpass

oname = 'bandpass_act_all_all_all.hdf5'
with h5py.File(opj(args.odir, oname), 'w') as hfile:

    for ar in arrays:

        hgroup = hfile.create_group(ar)
        hgroup.create_dataset('nu', data=bandpasses[ar]['nu'])
        hgroup.create_dataset('bandpass', data=bandpasses[ar]['bandpass'])
