from tacos import utils, data, models, mixing_matrix as M
from pixell import enmap, reproject
import healpy as hp
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
import argparse

parser = argparse.ArgumentParser(
    'Check that our mixing matrix, color correction, and unit conversion work by recovering PySM inputs.')
parser.add_argument('odir', type=str, help='Path to output dir')
parser.add_argument('--notes', dest='notes', type=str, default='', help='Notes to append to map names. ' +
                    'If notes not passed and tophat is passed, note will be "tophat"')
parser.add_argument('--tophat', dest='tophat', default=False, action='store_true', 
    help='If passed, make tophat pysm maps. Default is false: use the full corresponding instrument band. ' + 
    'If notes not passed and tophat is passed, note will be "tophat"')
args = parser.parse_args()

fig_path = args.odir
fig_path = fig_path + '/' if fig_path[-1] != '/' else fig_path
notes = args.notes
notes = 'tophat' if args.tophat and not notes else notes
tophat = args.tophat

# first get some Channels which we will compare to projected pysm inputs
instr_band = {}
instr_band['wmap'] = ['K']
instr_band['act'] = ['f090', 'f150', 'f220']
instr_band['planck'] = ['353']

pchannels = []
hchannels = []
for instr in instr_band:
    for band in instr_band[instr]:

        # now we want nu_sq_corr=True to match what pysm did (the default)
        pchannels.append(data.Channel(instr, band, pysm=True, healpix=False, pysm_notes=notes, bandpass_kwargs={'tophat': tophat}))
        hchannels.append(data.Channel(instr, band, pysm=True, healpix=True, pysm_notes=notes, bandpass_kwargs={'tophat': tophat}))

# next get our two components for comparison, synchrotron and dust
# to do this, first need inputs from pysm
import pysm3 
sky = pysm3.Sky(nside=512, preset_strings=['s1', 'd1'], output_unit='uK_CMB')
pysm_s, pysm_d = sky.components

nu0_s = pysm_s.freq_ref_P.value*1e9#np.array([pysm_s.freq_ref_I.value, pysm_s.freq_ref_P.value, pysm_s.freq_ref_P.value]).reshape(3,1)*1e9
beta_s = np.array(pysm_s.pl_index)

nu0_d = pysm_d.freq_ref_P.value*1e9#np.array([pysm_d.freq_ref_I.value, pysm_d.freq_ref_P.value, pysm_d.freq_ref_P.value]).reshape(3,1)*1e9
beta_d = np.array(pysm_d.mbb_index)
T_d = np.array(pysm_d.mbb_temperature)

hmodels = [models.Synch(nu0_s), models.Dust(nu0_d)]
hcomponents = [M.Component(hmodels[0], beta=beta_s), M.Component(hmodels[1], beta=beta_d, T=T_d)]

# get act geometry and project components to CAR
shape, wcs = enmap.read_map_geometry(utils.data_dir_str('raw', 'act') + 'map_pa4_f150_night_set0.fits')
shape = shape[-2:]

nu0_s_car = nu0_s#.reshape(3, 1, 1)
beta_s_car = reproject.enmap_from_healpix(beta_s, shape, wcs, rot=None)

nu0_d_car = nu0_d#.reshape(3, 1, 1)
beta_d_car = reproject.enmap_from_healpix(beta_d, shape, wcs, rot=None)
T_d_car = reproject.enmap_from_healpix(T_d, shape, wcs, rot=None)

pmodels = [models.Synch(nu0_s_car), models.Dust(nu0_d_car)]
pcomponents = [M.Component(pmodels[0], beta=beta_s_car), M.Component(pmodels[1], beta=beta_d_car, T=T_d_car)]

# get our amplitudes
a_s = np.array([pysm_s.I_ref.value, pysm_s.Q_ref.value, -pysm_s.U_ref.value])
a_d = np.array([pysm_d.I_ref.value, pysm_d.Q_ref.value, -pysm_d.U_ref.value])
ha = np.array([a_s, a_d])[:, 1:, ...]

a_s_car = reproject.enmap_from_healpix(a_s, shape, wcs, ncomp=3, rot=None)
a_d_car = reproject.enmap_from_healpix(a_d, shape, wcs, ncomp=3, rot=None)
pa = np.array([a_s_car, a_d_car])[:, 1:, ...]

pM = M.get_exact_mixing_matrix(pchannels, pcomponents, (2,) + shape, wcs=wcs)
hM = M.get_exact_mixing_matrix(hchannels, hcomponents, (2,) + T_d.shape)

print(ha.shape, hM.shape)
print(pa.shape, pM.shape)

# project into maps
hpmaps = np.einsum('jcax,cax->jax',hM,ha)
hrmaps = np.array([c.map for c in hchannels])[..., 1:, :]

ppmaps = np.einsum('jcayx,cayx->jayx',pM,pa)
prmaps = np.array([c.map for c in pchannels])[..., 1:, :, :]

if notes:
    notes += '_'

# plot comparisons
for j, channel in enumerate(hchannels):
    for a in range(2):
        instr = channel.instr
        band = channel.band
        pol = 'IQU'[a+1]
        hp.mollview((hpmaps - hrmaps)[j,a], unit='uK_CMB', title=f'Proj - PySM, {instr} {band}, {pol}')
        plt.savefig(fig_path + f'{instr}_{band}_{pol}_{notes}absdiff_healpix.png')
        plt.close()
        utils.eplot((ppmaps - prmaps)[j,a], colorbar=True, grid=False, fname=fig_path + f'{instr}_{band}_{pol}_{notes}absdiff')

for j, channel in enumerate(hchannels):
    for a in range(2):
        instr = channel.instr
        band = channel.band
        pol = 'IQU'[a+1]
        mean = 100*((hpmaps - hrmaps)/hpmaps)[j,a].mean()
        std = 100*((hpmaps - hrmaps)/hpmaps)[j,a].std()
        hp.mollview(100*((hpmaps - hrmaps)/hrmaps)[j,a], unit='%', title=f'(Proj - PySM)/Proj, {instr} {band}, {pol}', 
                    min=mean-std, max=mean+std)
        plt.savefig(fig_path + f'{instr}_{band}_{pol}_{notes}reldiff_healpix.png')
        plt.close()
        utils.eplot(100*((prmaps - ppmaps)/prmaps)[j,a], colorbar=True, grid=False, fname=fig_path + f'{instr}_{band}_{pol}_{notes}reldiff')

for j, channel in enumerate(hchannels):
    for a in range(2):
        instr = channel.instr
        band = channel.band
        pol = 'IQU'[a+1]
        print(instr, band)
        hdiff = np.abs(((hpmaps - hrmaps)/hrmaps)[j,a])
        hmaxdiff = np.max(hdiff)
        hmeandiff = np.mean(hdiff)
        hstdratio = np.std((hpmaps - hrmaps)[j,a]) / np.std(hrmaps[j,a])
        pdiff = np.abs(((ppmaps - prmaps)/prmaps)[j,a])
        pmaxdiff = np.max(pdiff)
        pmeandiff = np.mean(pdiff)
        pstdratio = np.std((ppmaps - prmaps)[j,a]) / np.std(prmaps[j,a])
        print(pol)
        print(f'maximum % difference -- healpix: {np.round(100*hmaxdiff, 5)}, car: {np.round(100*pmaxdiff, 5)}')
        print(f'mean % difference -- healpix: {np.round(100*hmeandiff, 5)}, car: {np.round(100*pmeandiff, 5)}')
        print(f'std ratio, % -- healpix: {np.round(100*hstdratio, 5)}, car: {np.round(100*pstdratio, 5)}')
        print(f'numpix % diff. >= 0.1% -- healpix: {np.nonzero(100*hdiff >= 0.1)[0].size}, car: {np.nonzero(100*pdiff >= 0.1)[0].size}')
        print('')
