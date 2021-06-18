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

parser = argparse.ArgumentParser('Take raw wmap beams and reduce/rename them for the tacos pipeline.')
args = parser.parse_args()

# get some basics
config = utils.config_from_yaml_resource('configs/data.yaml')
rawpath = utils.data_dir_str('raw', 'wmap') + 'beams/'
beampath = utils.data_dir_str('beam', 'wmap')

# define what we are looping over
freqs = ['K', 'Ka', 'Q', 'V', 'W']
beams = {}

# get act geometry
_, wcs = enmap.read_map_geometry(utils.data_dir_str('raw', 'act') + 'map_pa4_f150_night_set0.fits')
lmax = utils.lmax_from_wcs(wcs)

# for each freq, load raw beam and extend to lmax with a Gaussian
for freq in freqs:
    print(f'Working on {freq}')

    # get number of das
    if freq in ['K', 'Ka']:
        nda = 1
    elif freq in ['Q', 'V']:
        nda = 2
    else:
        nda = 4

    # load raw beam from disk
    for da in range(1, nda + 1):
        bell = np.loadtxt(rawpath + f'wmap_ampl_bl_{freq}{da}_9yr_v5p1.txt').T[1]

        # extend to lmax with Gaussian beam pegged at lmax_raw, b(lmax_raw)
        # if length of raw_data is 4001, then lmax_raw is 4000
        lmax_raw, blmax_raw = len(bell)-1, bell[-1]
        fwhm = utils.fwhm_from_ell_bell(lmax_raw, blmax_raw)
        gauss = hp.gauss_beam(fwhm, lmax=lmax)
        bell = np.append(bell, gauss[lmax_raw+1:])

        # save T beam as E, B beams
        bell = np.repeat(utils.atleast_nd(bell, 2), 3, axis=0)

        # add to data structure
        key = freq + str(da)
        beams[key] = {}
        beams[key]['ell'] = np.arange(lmax + 1)
        beams[key]['bell'] = bell

# save beams to disk
obeam_fn = utils.data_fn_str(type='beam', instr='wmap', band='all', id='all', set='all')

with h5py.File(beampath + obeam_fn, 'w') as hfile:

    for key in beams.keys():
        hgroup = hfile.create_group(key)
        hgroup.create_dataset('ell', data=beams[key]['ell'])
        hgroup.create_dataset('bell', data=beams[key]['bell'])