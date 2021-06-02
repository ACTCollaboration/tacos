#!/usr/bin/envs -u python3

# a script that only runs one and contains all the messiness to go from the raw
# data in tacos/raw to the reduced data.
#
# specific to act data. for now, just lumps arrays together. a modification to the mapsets could be made for specific
# detector maps later 

from pixell import enmap
from tacos import utils, data
import numpy as np

import argparse
parser = argparse.ArgumentParser('Take raw act-only GC maps and reduce/rename them for the tacos pipeline. In particular, \n \
coadd by common frequency. Default behavior of iau behavior argument may flip later.')
parser.add_argument('--diagonal-icovar', dest='diagonal_icovar', default=False, action='store_true',
    help='If passed, then neglect off-diagonal Stokes components in each pixel when coadding')
parser.add_argument('--single-set-coadd', dest='single_set_coadd', default=False, action='store_true',
    help='If passed, coadd any mapsets with single maps. This may introduce numerical errors.')
parser.add_argument('--raw-is-iau', dest='raw_is_iau', default=False, action='store_true',
    help='If passed, assume raw data already in IAU convention. Default is False: multiply U by -1')
args = parser.parse_args()

# get some basics
rawpath = data.config['raw_path'] + 'act/'
mappath = data.config['maps_path'] + 'act/'
covmatpath = data.config['covmats_path'] + 'act/'

# define the map/ivar name assignments
mapsets = dict(
map_f90_set0_fns = ['map_pa5_f090_night_set0.fits', 'map_pa6_f090_night_set0.fits'],
map_f90_set1_fns = ['map_pa5_f090_night_set1.fits', 'map_pa6_f090_night_set1.fits'],
map_f150_set0_fns = ['map_pa4_f150_night_set0.fits', 'map_pa5_f150_night_set0.fits', 'map_pa6_f150_night_set0.fits'],
map_f150_set1_fns = ['map_pa4_f150_night_set1.fits', 'map_pa5_f150_night_set1.fits', 'map_pa6_f150_night_set1.fits'],
map_f220_set0_fns = ['map_pa4_f220_night_set0.fits'],
map_f220_set1_fns = ['map_pa4_f220_night_set1.fits'],
)

mapsets.update(dict(
    map_f90_coadd_fns = mapsets['map_f090_set0_fns'] + mapsets['map_f090_set1_fns'],
    map_f150_coadd_fns = mapsets['map_f150_set0_fns'] + mapsets['map_f150_set1_fns'],
    map_f220_coadd_fns = mapsets['map_f220_set0_fns'] + mapsets['map_f220_set1_fns'],
))

ivarsets = {}
for k, v in mapsets.items():
    ivark = 'ivar' + k.split('map')[1]
    ivarv = []
    for mapfn in v:
        ivarv.append('div' + mapfn.split('map')[1])
    ivarsets[ivark] = ivarv

# define what we are looping over
# first do individual splits in same frequency, then do coadd of splits
freqs = ['f90', 'f150', 'f220']
splits = ['set0', 'set1', 'coadd']

# output info
if args.diagonal_icovar:
    print('Setting off-diagonal Stokes covariance (per pixel) to 0')
else:
    print('Preserving per-pixel off-diagonal Stokes covariance')

if not args.single_set_coadd:
    print('Single-map mapsets will not be coadded, just renamed')
else:
    print('Single-set mapsets will be coadded; this may introduce numerical errors')

if not args.raw_is_iau:
    print('Raw is not IAU. Multiplying U by -1 to get IAU.')
else:
    print('Raw is IAU. No correction to U needed.')

# for each combo, grab maps and ivars and get coadd
for freq in freqs:
    for split in splits:
        print(f'Working on {freq}, {split}')

        map_fns = mapsets[f'map_{freq}_{split}_fns']
        ivar_fns = ivarsets[f'ivar_{freq}_{split}_fns']
        imaps = [enmap.read_map(rawpath + m) for m in map_fns]
        icovars = [enmap.read_map(rawpath + m) for m in ivar_fns]

        print(map_fns)
        print(ivar_fns)

        imaps = enmap.ndmap(imaps, wcs=imaps[0].wcs) # shape is (nmap, npol, ny, nx)
        icovars = enmap.ndmap(icovars, wcs=icovars[0].wcs) # shape is (nmap, npol, npol, ny, nx)

        # get iau behavior
        if not args.raw_is_iau:
            imaps[..., 2, :, :] *= -1
                        
            # correct for iau convention on U pol crosses
            for i in range(icovars.shape[-4]):
                for j in range(i+1, icovars.shape[-3]): # off-diagonals only
                    if (i, j) == (0, 2) or (i, j) == (1, 2):
                        icovars[:, i, j] *= -1
                        icovars[:, j, i] *= -1

        # get diagonal version of icovars, if passed
        if args.diagonal_icovar:
            icovars = np.diagonal(icovars, axis1=-4, axis2=-3)

            # just reshaping to broadcast against original icovar shape
            N = icovars.shape[-1]
            icovars = np.einsum('ab,...b->ab...', np.eye(N, dtype=int), icovars)
            icovars = np.moveaxis(icovars, (0, 1), (-4, -3))
            icovars = enmap.samewcs(icovars, imaps.wcs)    

        # separate out the monopoles
        imaps_monopole = np.mean(imaps, axis=(-2, -1), keepdims=True)
        imaps_monopole = np.broadcast_to(imaps_monopole, imaps.shape, subok=True)
        imaps_zero_monopole = imaps - imaps_monopole

        # symmetrize the covars
        icovars = utils.symmetrize(icovars, axis1=-4, axis2=-3)

        # coadd
        map_coadd, icovar_coadd = utils.get_coadd_map_covar(imaps_zero_monopole, icovars, return_icovar_coadd=True)
        monopole_coadd = utils.get_coadd_map_covar(imaps_monopole, icovars)

        # overwrite output for single-map mapsets if necessary
        if not args.single_set_coadd:
            if imaps.shape[0] == 1:
                print('Skipping coadd of single-map mapset')
                map_coadd = imaps_zero_monopole[0]

        # save
        extra = {'POLCCONV': 'IAU'}

        omap_fn = data.data_str(type='map', instr='act', band=freq, id='all', set=split)
        #enmap.write_map(mappath + omap_fn, map_coadd, extra=extra)

        ocoadd_fn = data.data_str(type='icovar', instr='act', band=freq, id='all', set=split)
        #enmap.write_map(covmatpath + ocoadd_fn, icovar_coadd, extra=extra)