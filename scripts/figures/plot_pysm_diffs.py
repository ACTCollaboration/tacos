from tacos import utils, bandpass, data, sampling, models
from pixell import enmap, enplot, reproject
import healpy as hp
import matplotlib.pyplot as plt

import numpy as np
import argparse

parser = argparse.ArgumentParser(
    'Check that our mixing matrix, color correction, and unit conversion work by recovering PySM inputs.')
parser.add_argument('odir', type=str, help='Path to output dir')
args = parser.parse_args()

fig_path = args.odir

# first get some Channels which we will compare to projected pysm inputs
instr_band = {}
instr_band['wmap'] = ['K']
instr_band['act'] = ['f090', 'f150', 'f220']
instr_band['planck'] = ['353']

pchannels = []
hchannels = []
for instr in instr_band:
    for band in instr_band[instr]:
        pchannels.append(data.Channel(instr, band, pysm=True, healpix=False))
        hchannels.append(data.Channel(instr, band, pysm=True, healpix=True))

# next get our two components for comparison, synchrotron and dust
# to do this, first need inputs from pysm
import pysm3 
sky = pysm3.Sky(nside=512, preset_strings=['s1', 'd1'], output_unit='uK_CMB')
pysm_s, pysm_d = sky.components

nu0_s = np.array([pysm_s.freq_ref_I.value, pysm_s.freq_ref_P.value, pysm_s.freq_ref_P.value]).reshape(3,1)*1e9
beta_s = np.array(pysm_s.pl_index)

nu0_d = np.array([pysm_d.freq_ref_I.value, pysm_d.freq_ref_P.value, pysm_d.freq_ref_P.value]).reshape(3,1)*1e9
beta_d = np.array(pysm_d.mbb_index)
T_d = np.array(pysm_d.mbb_temperature)

hcomponents = [models.Synch(nu0_s, beta=beta_s), models.Dust(nu0_d, beta=beta_d, T=T_d)]

# get act geometry and project components to CAR
shape, wcs = enmap.read_map_geometry(utils.data_dir_str('raw', 'act') + 'map_pa4_f150_night_set0.fits')
shape = shape[-2:]

nu0_s_car = nu0_s.reshape(3, 1, 1)
beta_s_car = reproject.enmap_from_healpix(beta_s, shape, wcs, rot=None)

nu0_d_car = nu0_d.reshape(3, 1, 1)
beta_d_car = reproject.enmap_from_healpix(beta_d, shape, wcs, rot=None)
T_d_car = reproject.enmap_from_healpix(T_d, shape, wcs, rot=None)

pcomponents = [models.Synch(nu0_s_car, beta=beta_s_car), models.Dust(nu0_d_car, beta=beta_d_car, T=T_d_car)]

# get our mixing matrices
pM = sampling.get_mixing_matrix(pchannels, pcomponents)
hM = sampling.get_mixing_matrix(hchannels, hcomponents)

# get the amplitudes to project with our matrix from pysm
# include minus sign for IAU
a_s = np.array([pysm_s.I_ref.value, pysm_s.Q_ref.value, -pysm_s.U_ref.value])
a_d = np.array([pysm_d.I_ref.value, pysm_d.Q_ref.value, -pysm_d.U_ref.value])
ha = np.array([a_s, a_d])

# project amplitudes to CAR
a_s_car = utils.atleast_nd(reproject.enmap_from_healpix(a_s, shape, wcs, ncomp=3, rot=None), 4)
a_d_car = utils.atleast_nd(reproject.enmap_from_healpix(a_d, shape, wcs, ncomp=3, rot=None), 4)
pa = np.array([a_s_car, a_d_car])

# project into maps
hpmaps = np.einsum('jciax,cax->jiax',hM,ha)
hrmaps = np.array([c.map for c in hchannels])

ppmaps = np.einsum('jcmayx,cmayx->jmayx',pM,pa)
prmaps = np.array([c.map for c in pchannels])

# plot comparisons
for j, channel in enumerate(hchannels):
    for a in range(3):
        instr = channel.instr
        band = channel.band
        pol = 'IQU'[a]
        hp.mollview((hpmaps - hrmaps)[j,0,a], unit='uK_CMB', title=f'Proj - PySM, {instr} {band}, {pol}')
        plt.savefig(fig_path + f'{instr}_{band}_{pol}_absdiff_healpix.png')
        plt.close()
        utils.eplot((ppmaps - prmaps)[j,0,a], colorbar=True, grid=False, fname=fig_path + f'{instr}_{band}_{pol}_absdiff')

for j, channel in enumerate(hchannels):
    for a in range(3):
        instr = channel.instr
        band = channel.band
        pol = 'IQU'[a]
        mean = 100*((hpmaps - hrmaps)/hpmaps)[j,0,a].mean()
        std = 100*((hpmaps - hrmaps)/hpmaps)[j,0,a].std()
        hp.mollview(100*((hpmaps - hrmaps)/hrmaps)[j,0,a], unit='%', title=f'(Proj - PySM)/Proj, {instr} {band}, {pol}', 
                    min=mean-std, max=mean+std)
        plt.savefig(fig_path + f'{instr}_{band}_{pol}_reldiff_healpix.png')
        plt.close()
        utils.eplot(100*((prmaps - ppmaps)/prmaps)[j,0,a], colorbar=True, grid=False, fname=fig_path + f'{instr}_{band}_{pol}_reldiff')


for j, channel in enumerate(hchannels):
    for a in range(3):
        instr = channel.instr
        band = channel.band
        pol = 'IQU'[a]
        print(instr, band)
        hdiff = np.abs(((hpmaps - hrmaps)/hrmaps)[j,0,a])
        hmaxdiff = np.max(hdiff)
        hmeandiff = np.mean(hdiff)
        hstdratio = np.std((hpmaps - hrmaps)[j,0,a]) / np.std(hrmaps[j,0,a])
        pdiff = np.abs(((ppmaps - prmaps)/prmaps)[j,0,a])
        pmaxdiff = np.max(pdiff)
        pmeandiff = np.mean(pdiff)
        pstdratio = np.std((ppmaps - prmaps)[j,0,a]) / np.std(prmaps[j,0,a])
        print(pol)
        print(f'maximum % difference -- healpix: {np.round(100*hmaxdiff, 5)}, car: {np.round(100*pmaxdiff, 5)}')
        print(f'mean % difference -- healpix: {np.round(100*hmeandiff, 5)}, car: {np.round(100*pmeandiff, 5)}')
        print(f'std ratio, % -- healpix: {np.round(100*hstdratio, 5)}, car: {np.round(100*pstdratio, 5)}')
        print(f'numpix % diff. >= 0.1% -- healpix: {np.nonzero(100*hdiff >= 0.1)[0].size}, car: {np.nonzero(100*pdiff >= 0.1)[0].size}')
        print('')
