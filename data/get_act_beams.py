#!/usr/bin/envs -u python3

# a script that only runs one and contains all the messiness to go from the raw
# data in tacos/raw to the reduced data.
#
# specific to planck data. a modification to the mapsets could be made for specific
# detector maps later 

from tacos import utils, data

from pixell import enmap
import healpy as hp
from astropy.io import fits
from astropy.table import Table
import h5py
import numpy as np

import argparse

parser = argparse.ArgumentParser('Take raw act beams and reduce/rename them for the tacos pipeline.')
args = parser.parse_args()

# get some basics
rawpath = data.config['raw_path'] + 'act/beams/'
beampath = data.config['beams_path'] + 'act/'

# define what we are looping over
arrays = ['pa4_f150','pa4_f220', 'pa5_f090', 'pa5_f150', 'pa6_f090', 'pa6_f150']
beams = {}

# get act geometry
_, wcs = enmap.read_map_geometry(data.config['raw_path'] + 'act/map_pa4_f150_night_set0.fits')
lmax = utils.lmax_from_wcs(wcs)

# for each freq, load raw beam and truncate at lmax
for ar in arrays:
    print(f'Working on {ar}')

    # load raw beam from disk
    bell = np.loadtxt(rawpath + f'b20190504_s17_{ar}_nohwp_night_beam_tform_instant.txt').T[1]

    # truncate when normalized beam falls below 1e-4
    bell = bell/np.max(bell)
    lsplice = np.argmin(np.abs(bell - 1e-4))
    bell = bell[:lsplice]

    # extend to lmax with Gaussian beam pegged at lmax_raw, b(lmax_raw)
    # if length of raw_data is 4001, then lmax_raw is 4000
    # only do this if lsplice is less leq lmax
    if len(bell) < lmax + 1:
        lmax_raw, blmax_raw = len(bell)-1, bell[-1]
        fwhm = utils.fwhm_from_ell_bell(lmax_raw, blmax_raw)
        gauss = hp.gauss_beam(fwhm, lmax=lmax)
        bell = np.append(bell, gauss[lmax_raw+1:])
    else:
        print(f'Skipping Gaussian extension, lsplice is {lsplice}, lmax is {lmax}')

    # add to data structure
    beams[ar] = {}
    beams[ar]['ell'] = np.arange(lmax + 1)
    beams[ar]['bell'] = bell

# save beams to disk
obeam_fn = data.data_str(type='beam', instr='act', band='all', id='all', set='all')

with h5py.File(beampath + obeam_fn, 'w') as hfile:

    for ar in beams.keys():
        hgroup = hfile.create_group(ar)
        hgroup.create_dataset('ell', data=beams[ar]['ell'])
        hgroup.create_dataset('bell', data=beams[ar]['bell'])