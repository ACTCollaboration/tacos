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

parser = argparse.ArgumentParser('Take raw planck beams and reduce/rename them for the tacos pipeline.')
parser.add_argument('--T-only', dest='T_only', default=False, action='store_true',
    help='Only use the T beam in place of the polarization beams')
args = parser.parse_args()

# get some basics
config = utils.config_from_yaml_resource('configs/data.yaml')
rawpath = utils.data_dir_str('raw', 'planck') + 'beams/BeamWf_HFI_R3.01/'
beampath = utils.data_dir_str('beam', 'planck')

# define what we are looping over
freqs = ['100', '143', '217', '353', '353_psb']
beams = {}

# get act geometry
_, wcs = enmap.read_map_geometry(utils.data_dir_str('raw', 'act') + 'map_pa4_f150_night_set0.fits')
lmax = utils.lmax_from_wcs(wcs)

# for each freq, load raw beam and extend to lmax with a Gaussian
for freq in freqs:
    print(f'Working on {freq}')

    # correct for psb filename if necessary
    if '_psb' in freq:
        ifreq = freq.split('_psb')[0] + 'p'
    else:
        ifreq = freq 

    # load raw beam from disk
    beam_fn = f'Bl_TEB_R3.01_fullsky_{ifreq}x{ifreq}.fits'
    
    with fits.open(rawpath + beam_fn) as hdul:
        idata = Table(hdul[1].data)

    if args.T_only:
        bell = np.repeat(utils.atleast_nd(idata['T'], 2), 3, axis=0)
    else:
        bell = np.array([idata['T'], idata['E'], idata['B']])

    # extend to lmax with Gaussian beam pegged at lmax_raw, b(lmax_raw)
    # if length of raw_data is 4001, then lmax_raw is 4000
    extended_bell = []
    for b in bell:
        lmax_raw, blmax_raw = len(b)-1, b[-1]
        fwhm = utils.fwhm_from_ell_bell(lmax_raw, blmax_raw)
        gauss = hp.gauss_beam(fwhm, lmax=lmax)
        extended_bell.append(np.append(b, gauss[lmax_raw+1:]))

    # add to data structure
    beams[freq] = {}
    beams[freq]['ell'] = np.arange(lmax + 1)
    beams[freq]['bell'] = np.array(extended_bell)

# save beams to disk
obeam_fn = utils.data_fn_str(type='beam', instr='planck', band='all', id='all', set='all')

with h5py.File(beampath + obeam_fn, 'w') as hfile:

    for freq in freqs:
        hgroup = hfile.create_group(freq)
        hgroup.create_dataset('ell', data=beams[freq]['ell'])
        hgroup.create_dataset('bell', data=beams[freq]['bell'])