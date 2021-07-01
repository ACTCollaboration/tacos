#!/usr/bin/envs -u python3

# a script that only runs one and contains all the messiness to get a raw pysm map
#
# generates pysm maps at each band specified in configs/bandpass_config.yaml 

from pixell import enmap, reproject
import healpy as hp 
import pysm3 

from tacos import utils, data
from tacos.bandpass import BandPass
import numpy as np

import argparse
parser = argparse.ArgumentParser('Generate pysm maps at each band in this project. \
    Default behavior of iau behavior argument may flip later.')
parser.add_argument('--raw-is-iau', dest='raw_is_iau', default=False, action='store_true',
    help='If passed, assume raw data already in IAU convention. Default is False: multiply U by -1')
parser.add_argument('--odtype', dest='odtype', type=str, default='f4', help='Numpy dtype str to apply to written products.')
args = parser.parse_args()

# get some basics
mappath = utils.data_dir_str('map', 'pysm')

# define the map/ivar name assignments
instr_band = {}
instr_band['act'] = ['f090', 'f150', 'f220']
instr_band['planck'] = ['100', '143', '217', '353']
instr_band['wmap'] = ['K', 'Ka', 'Q', 'V', 'W']

# print helper statements
if not args.raw_is_iau:
    print('Raw is not IAU. Multiplying U by -1 to get IAU.')
else:
    print('Raw is IAU. No correction to U needed.')

# get act geometry
shape, wcs = enmap.read_map_geometry(utils.data_dir_str('raw', 'act') + 'map_pa4_f150_night_set0.fits')
shape = shape[-2:]

# get sky object
sky = pysm3.Sky(nside=512, preset_strings=['s1', 'd1'], output_unit='uK_CMB')

# for each combo, grab maps and ivars and get coadd
for instr in instr_band:
    for band in instr_band[instr]:
        print(f'Working on {instr}, {band}')

        # load the pysm bandpass
        bandpass_path = utils.data_dir_str('bandpass', instr)
        bandpass_path += utils.data_fn_str(type='bandpass', instr=instr, band='all', id='all', set='all')
        
        # pysm applies nu**2 to convert to Jy/sr from RJ, so we need to do the same (supply default nu_sq_corr=True)
        bandpass = BandPass.load_pysm_bandpass(bandpass_path, instr, band)

        # get the healpix map
        hmap = sky.get_emission(bandpass.nu * pysm3.units.Hz)

        # project map to act patch using SHT interpolation
        print('SHT interpolation of map')
        imap = reproject.enmap_from_healpix(hmap[:3], shape, wcs, ncomp=3, rot=None) # shape is (npol, ny, nx)

        # get iau behavior
        if not args.raw_is_iau:
            hmap[..., 2, :] *= -1
            imap[..., 2, :, :] *= -1

        # save
        extra = {'POLCCONV': 'IAU'}

        hmap_fn = utils.data_fn_str(type='map', instr='pysm', band=band, id='all', set='all', notes='healpix')
        hp.write_map(mappath + hmap_fn, hmap.astype(args.odtype), extra_header=extra, overwrite=True)

        imap_fn = utils.data_fn_str(type='map', instr='pysm', band=band, id='all', set='all')
        enmap.write_map(mappath + imap_fn, imap.astype(args.odtype), extra=extra)