from tacos import utils, data, models, mixing_matrix as M
from pixell import enmap, reproject
import healpy as hp
import matplotlib.pyplot as plt

import numpy as np
import argparse

import pysm3

parser = argparse.ArgumentParser(
    'Check that our mixing matrix, color correction, and unit conversion work by recovering PySM inputs.')
parser.add_argument('odir', type=str, help='Path to output dir')
parser.add_argument('--odtype', dest='odtype', type=str, default='f4', help='Numpy dtype str to apply to written products.')
args = parser.parse_args()

# next get our components: synchrotron and dust
# to do this, first need inputs from pysm
# although pysm will udgrade to the right resolution, we want the "true" raw data, which is 
# nside=512, see: https://github.com/healpy/pysm/blob/master/pysm3/data/presets.cfg and https://portal.nersc.gov/project/cmb/pysm-data/pysm_2/
sky = pysm3.Sky(nside=512, preset_strings=['s1', 'd1'], output_unit='uK_CMB')
pysm_s, pysm_d = sky.components

beta_s = np.array(pysm_s.pl_index)

beta_d = np.array(pysm_d.mbb_index)
T_d = np.array(pysm_d.mbb_temperature)

# get act geometry and project components to CAR
shape, wcs = enmap.read_map_geometry(utils.data_dir_str('raw', 'act') + 'map_pa4_f150_night_set0.fits')
shape = shape[-2:]

beta_s_car = reproject.enmap_from_healpix(beta_s, shape, wcs, rot=None)

beta_d_car = reproject.enmap_from_healpix(beta_d, shape, wcs, rot=None)
T_d_car = reproject.enmap_from_healpix(T_d, shape, wcs, rot=None)

# save our templates to disk
hp.write_map(args.odir + 'pysm_Synch_beta_healpix.fits', beta_s, dtype=args.odtype, coord='G', overwrite=True)
hp.write_map(args.odir + 'pysm_Dust_beta_healpix.fits', beta_d, dtype=args.odtype, coord='G', overwrite=True)
hp.write_map(args.odir + 'pysm_Dust_T_healpix.fits', beta_d, dtype=args.odtype, coord='G', column_units='K', overwrite=True)

enmap.write_map(args.odir + 'pysm_Synch_beta.fits', beta_s_car)
enmap.write_map(args.odir + 'pysm_Dust_beta.fits', beta_d_car)
enmap.write_map(args.odir + 'pysm_Dust_T.fits', T_d_car)
