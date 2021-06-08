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
parser.add_argument('--component', dest='component', type=str, default='T', help='Polarization component to grab from disk')
args = parser.parse_args()

# get some basics
rawpath = data.config['raw_path'] + 'planck/beams/BeamWf_HFI_R3.01/'
beampath = data.config['beams_path'] + 'planck/'

# define what we are looping over
freqs = ['100', '143', '217', '353', '353_psb']
beams = {}

# get act geometry
_, wcs = enmap.read_map_geometry(data.config['raw_path'] + 'act/map_pa4_f150_night_set0.fits')
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

    bell = idata[args.component]

    # extend to lmax with Gaussian beam pegged at lmax_raw, b(lmax_raw)
    # if length of raw_data is 4001, then lmax_raw is 4000
    lmax_raw, blmax_raw = len(bell)-1, bell[-1]
    fwhm = utils.fwhm_from_ell_bell(lmax_raw, blmax_raw)
    gauss = hp.gauss_beam(fwhm, lmax=lmax)
    bell = np.append(bell, gauss[lmax_raw+1:])

    # add to data structure
    beams[freq] = {}
    beams[freq]['ell'] = np.arange(lmax + 1)
    beams[freq]['bell'] = bell

# save beams to disk
obeam_fn = data.data_str(type='beam', instr='planck', band='all', id='all', set='all')

with h5py.File(beampath + obeam_fn, 'w') as hfile:

    for freq in freqs:
        hgroup = hfile.create_group(freq)
        hgroup.create_dataset('ell', data=beams[freq]['ell'])
        hgroup.create_dataset('bell', data=beams[freq]['bell'])