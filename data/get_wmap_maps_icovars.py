#!/usr/bin/envs -u python3

# a script that only runs one and contains all the messiness to go from the raw
# data in tacos/raw to the reduced data.
#
# specific to wmap data. a modification to the mapsets could be made for specific
# detector maps later 

from pixell import enmap, reproject
import healpy as hp
from tacos import utils, data
import numpy as np

import argparse
parser = argparse.ArgumentParser('Take raw wmap maps and reduce/rename them for the tacos pipeline. In particular, \n \
project to act GC patch and fix IAU convention.')
parser.add_argument('--diagonal-icovar', dest='diagonal_icovar', default=False, action='store_true',
    help='If passed, then neglect off-diagonal Stokes components in each pixel when coadding')
parser.add_argument('--single-set-coadd', dest='single_set_coadd', default=False, action='store_true',
    help='If passed, coadd any mapsets with single maps. This may introduce numerical errors.')
parser.add_argument('--odtype', dest='odtype', type=str, default='f4', help='Numpy dtype str to apply to written products.')
args = parser.parse_args()

# get some basics
rawpath = data.config['raw_path'] + 'wmap/'
mappath = data.config['maps_path'] + 'wmap/'
covmatpath = data.config['covmats_path'] + 'wmap/'

# define the map/ivar name assignments
mapsets = dict(
)

mapsets.update(dict(
    map_K_coadd_fns = ['wmap_band_iqumap_r9_9yr_K_v5.fits'],
    map_Ka_coadd_fns = ['wmap_band_iqumap_r9_9yr_Ka_v5.fits'],
    map_Q_coadd_fns = ['wmap_band_iqumap_r9_9yr_Q_v5.fits'],
    map_V_coadd_fns = ['wmap_band_iqumap_r9_9yr_V_v5.fits'],
    map_W_coadd_fns = ['wmap_band_iqumap_r9_9yr_W_v5.fits'],
))

# define what we are looping over
# first do individual splits in same frequency, then do coadd of splits
freqs = ['K', 'Ka', 'Q', 'V', 'W']
sig2_temp = (np.array([1.429, 1.466, 2.188, 3.131, 6.544])*1e3)**2
sig2_pol = (np.array([1.435, 1.472, 2.197, 3.141, 6.560])*1e3)**2
splits = ['coadd']

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
        f = freqs.index(freq)

        map_fns = mapsets[f'map_{freq}_{split}_fns']
        wdata_maps = [hp.read_map(rawpath + m, field=None, hdu=1) for m in map_fns]
        wdata_nobs = [hp.read_map(rawpath + m, field=None, hdu=2) for m in map_fns]

        print(map_fns)

        # project map to act patch using SHT interpolation
        print('SHT interpolation of map')
        imaps = [reproject.enmap_from_healpix(w[:3], shape, wcs, ncomp=3, unit=1e-3, rot=None) for w in wdata_maps]
        imaps = enmap.ndmap(imaps, wcs=imaps[0].wcs) # shape is (nmap, npol, ny, nx)

        # correct for iau convention
        imaps[..., 2, :, :] *= -1

        # project icovars to act patch using nearest-neighbor interpolation
        print('Nearest-neighbor interpolation of icovar')
        oshape = (len(wdata_nobs), imaps.shape[-3]) + imaps.shape[-3:] # shape is (nmap, npol, npol, ny, nx)
        icovars = enmap.zeros(oshape, wcs)
        for i, w in enumerate(wdata_nobs):
            nobs_w = np.array([reproject.enmap_from_healpix_interp(w[j], shape, wcs, rot=None) for j in range(4)])
            icovars_w = np.zeros(oshape[1:]) # on per-map basis

            # fill inverse-covariance matrix with 1/uK^2 entries
            for j, comp in enumerate([(0,0), (1,1), (1,2), (2,2)]):
                m, n = comp
                if j == 0:
                    sig2 = sig2_temp[f]
                else:
                    sig2 = sig2_pol[f]
                icovars_w[m, n] = nobs_w[j] / sig2
            
            # store data
            icovars[i] = icovars_w
    
        # correct for iau convention on U pol crosses
        for i in range(icovars.shape[-4]):
            for j in range(i+1, icovars.shape[-3]): # off-diagonals only
                if (i, j) == (0, 2) or (i, j) == (1, 2):
                    icovars[:, i, j] *= -1

        # symmetrize the covars
        icovars = utils.symmetrize(icovars, axis1=-4, axis2=-3, method='from_triu')

        # weight by pixel area factors
        act_pixsize = enmap.pixsizemap(shape, wcs)
        assert np.unique([w[0].size for w in wdata_nobs]).size == 1 # make sure all maps have same nside
        hp_pixsize = hp.nside2pixarea(hp.npix2nside(wdata_nobs[0][0].size))
        icovars *= (act_pixsize / hp_pixsize)

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

        # coadd
        map_coadd, icovar_coadd = utils.get_coadd_map_icovar(imaps_zero_monopole, icovars, return_icovar_coadd=True)
        monopole_coadd = utils.get_coadd_map_icovar(imaps_monopole, icovars)

        # overwrite output for single-map mapsets if necessary
        if not args.single_set_coadd:
            if imaps.shape[0] == 1:
                print('Skipping coadd of single-map mapset')
                map_coadd = imaps_zero_monopole[0]

        # save
        extra = {'POLCCONV': 'IAU'}

        omap_fn = data.data_str(type='map', instr='wmap', band=freq, id='all', set=split)
        enmap.write_map(mappath + omap_fn, map_coadd.astype(args.odtype), extra=extra)

        ocoadd_fn = data.data_str(type='icovar', instr='wmap', band=freq, id='all', set=split)
        enmap.write_map(covmatpath + ocoadd_fn, icovar_coadd.astype(args.odtype), extra=extra)