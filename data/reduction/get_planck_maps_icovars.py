#!/usr/bin/envs -u python3

# a script that only runs one and contains all the messiness to go from the raw
# data in tacos/raw to the reduced data.
#
# specific to planck data. a modification to the mapsets could be made for specific
# detector maps later 

from pixell import enmap, reproject
import healpy as hp
from tacos import utils, data
import numpy as np

import argparse
parser = argparse.ArgumentParser('Take raw planck maps and reduce/rename them for the tacos pipeline. In particular, \n \
project to act GC patch and fix IAU convention.')
parser.add_argument('--diagonal-icovar', dest='diagonal_icovar', default=False, action='store_true',
    help='If passed, then neglect off-diagonal Stokes components in each pixel when coadding')
parser.add_argument('--single-set-coadd', dest='single_set_coadd', default=False, action='store_true',
    help='If passed, coadd any mapsets with single maps. This may introduce numerical errors.')
parser.add_argument('--odtype', dest='odtype', type=str, default='f4', help='Numpy dtype str to apply to written products.')
args = parser.parse_args()

# get some basics
rawpath = data.config['raw_path'] + 'planck/'
mappath = data.config['maps_path'] + 'planck/'
covmatpath = data.config['covmats_path'] + 'planck/'

# define the map/ivar name assignments
mapsets = dict(
map_100_set0_fns = ['HFI_SkyMap_100-1-4_2048_R4.00_full.fits'],
map_100_set1_fns = ['HFI_SkyMap_100-2-3_2048_R4.00_full.fits'],
map_143_set0_fns = ['HFI_SkyMap_143-1-3-5-7_2048_R4.00_full.fits'],
map_143_set1_fns = ['HFI_SkyMap_143-2-4-6_2048_R4.00_full.fits'],
map_217_set0_fns = ['HFI_SkyMap_217-1-3-5-7_2048_R4.00_full.fits'],
map_217_set1_fns = ['HFI_SkyMap_217-2-4-6-8_2048_R4.00_full.fits'],
map_353_set0_fns = ['HFI_SkyMap_353-1-3-5-7_2048_R4.00_full.fits'],
map_353_set1_fns = ['HFI_SkyMap_353-2-4-6-8_2048_R4.00_full.fits'],
map_545_set0_fns = ['HFI_SkyMap_545-1_2048_R4.00_full.fits'],
map_545_set1_fns = ['HFI_SkyMap_545-2-4_2048_R4.00_full.fits'],
map_857_set0_fns = ['HFI_SkyMap_857-1-3_2048_R4.00_full.fits'],
map_857_set1_fns = ['HFI_SkyMap_857-2-4_2048_R4.00_full.fits'],
)

mapsets.update(dict(
    map_100_coadd_fns = mapsets['map_100_set0_fns'] + mapsets['map_100_set1_fns'],
    map_143_coadd_fns = mapsets['map_143_set0_fns'] + mapsets['map_143_set1_fns'],
    map_217_coadd_fns = mapsets['map_217_set0_fns'] + mapsets['map_217_set1_fns'],
    map_353_coadd_fns = mapsets['map_353_set0_fns'] + mapsets['map_353_set1_fns'],
    map_545_coadd_fns = mapsets['map_545_set0_fns'] + mapsets['map_545_set1_fns'],
    map_857_coadd_fns = mapsets['map_857_set0_fns'] + mapsets['map_857_set1_fns'],
))

# define what we are looping over
# first do individual splits in same frequency, then do coadd of splits
freqs = ['100', '143', '217', '353']
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

print(f'Writing to dtype={args.odtype}')

# get act geometry
shape, wcs = enmap.read_map_geometry(data.config['raw_path'] + 'act/map_pa4_f150_night_set0.fits')
shape = shape[-2:]

# for each combo, grab maps and ivars and get coadd
for freq in freqs:
    for split in splits:
        print(f'Working on {freq}, {split}')

        map_fns = mapsets[f'map_{freq}_{split}_fns']
        pdata = [hp.read_map(rawpath + m, field=None) for m in map_fns]

        print(map_fns)

        # project map to act patch using SHT interpolation
        print('SHT interpolation of map')
        imaps = [reproject.enmap_from_healpix(p[:3], shape, wcs, ncomp=3, unit=1e-6, rot=None) for p in pdata]
        imaps = enmap.ndmap(imaps, wcs=imaps[0].wcs) # shape is (nmap, npol, ny, nx)

        # correct for iau convention
        imaps[..., 2, :, :] *= -1

        # project icovars to act patch using nearest-neighbor interpolation
        print('Nearest-neighbor interpolation of icovar')
        oshape = (len(pdata), imaps.shape[-3]) + imaps.shape[-3:] # shape is (nmap, npol, npol, ny, nx)
        icovars = enmap.zeros(oshape, wcs)
        for i, p in enumerate(pdata):
            icovars_p = np.array([reproject.enmap_from_healpix_interp(p[j], shape, wcs, rot=None)/1e-6 for j in range(4, 10)])
            icovars[i][np.triu_indices(imaps.shape[-3])] = icovars_p
        
        # weight by pixel area factors
        act_pixsize = enmap.pixsizemap(shape, wcs)
        assert np.unique([p[0].size for p in pdata]).size == 1 # make sure all maps have same nside
        hp_pixsize = hp.nside2pixarea(hp.npix2nside(pdata[0][0].size))
        icovars *= (act_pixsize / hp_pixsize)
    
        # correct for iau convention on U pol crosses
        for i in range(icovars.shape[-4]):
            for j in range(i+1, icovars.shape[-3]): # off-diagonals only
                if (i, j) == (0, 2) or (i, j) == (1, 2):
                    icovars[:, i, j] *= -1

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
        icovars = utils.symmetrize(icovars, axis1=-4, axis2=-3, method='from_triu')

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

        omap_fn = data.data_str(type='map', instr='planck', band=freq, id='all', set=split)
        enmap.write_map(mappath + omap_fn, map_coadd.astype(args.odtype), extra=extra)

        ocoadd_fn = data.data_str(type='icovar', instr='planck', band=freq, id='all', set=split)
        enmap.write_map(covmatpath + ocoadd_fn, icovar_coadd.astype(args.odtype), extra=extra)