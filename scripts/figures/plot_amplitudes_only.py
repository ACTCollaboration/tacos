from tacos import utils, chain, mixing_matrix
from tacos.utils import eigpow

from pixell import enmap, reproject
import pysm3

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser('Generate an example chain of only amplitudes',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('config_path', type=str, help='Path to run config, absolute or relative to tacos')
args = parser.parse_args()

# make some interesting plots 
best = chain.get_all_amplitudes().mean(axis=0)

# flatten component/pol axes to iterate over them
m = enmap.samewcs(m.reshape(-1, 480, 960), best)
MN_invM_INV = eigpow(MN_invM.reshape(ncomp*npol, ncomp*npol, *MN_invM.shape[-2:]), -1, axes=(0, 1))
MN_invM_INV = MN_invM_INV.reshape(ncomp, npol, ncomp, npol, *MN_invM_INV.shape[-2:])
Md = np.einsum('cadb...,ca...->db...', MN_invM_INV, MN_invd)
Md = enmap.samewcs(Md.reshape(-1, 480, 960), best)

pm = best.reshape(-1, 480, 960) - m
pMd = best.reshape(-1, 480, 960) - Md
for i in range(len(best.reshape(-1, 480, 960))):
    comp = ['dust', 'synch'][i//npol]
    pol = polstr[i%npol]
    if prior:
        utils.eshow(m[i], colorbar=True, fname=f'/scratch/gpfs/zatkins/data/ACTCollaboration/tacos/examples/prior_{comp}_{pol}_{chain.name}')
        utils.eshow(pm[i], colorbar=True, fname=f'/scratch/gpfs/zatkins/data/ACTCollaboration/tacos/examples/best-prior_{comp}_{pol}_{chain.name}')
    else:
        utils.eshow(Md[i], colorbar=True, fname=f'/scratch/gpfs/zatkins/data/ACTCollaboration/tacos/examples/data_{comp}_{pol}_{chain.name}')
        utils.eshow(pMd[i], colorbar=True, fname=f'/scratch/gpfs/zatkins/data/ACTCollaboration/tacos/examples/best-data_{comp}_{pol}_{chain.name}')
        utils.eshow(enmap.smooth_gauss(Md[i], np.radians(10/60) / np.sqrt(8 * np.log(2))), colorbar=True, fname=f'/scratch/gpfs/zatkins/data/ACTCollaboration/tacos/examples/data_smoothed_{comp}_{pol}_{chain.name}')
        utils.eshow(enmap.smooth_gauss(pMd[i], np.radians(10/60) / np.sqrt(8 * np.log(2))), colorbar=True, fname=f'/scratch/gpfs/zatkins/data/ACTCollaboration/tacos/examples/best-data_smoothed_{comp}_{pol}_{chain.name}')

mychi2 = chi2_per_pix(chain.get_all_amplitudes(), mean=best)
fig = plt.figure(figsize=(8, 6))
_, bins, _ = plt.hist(mychi2.reshape(-1), bins=200, histtype='step', label='samples', density=True)
x = np.linspace(0,bins[-1], 1000)
plt.plot(x, chi2.pdf(x, df=ncomp*npol), linestyle='--', label=rf'$\chi^2_{ncomp*npol}$')
plt.xlim(0, 20)
plt.legend()
type = 'prior' if prior else 'data'
plt.savefig(f'/scratch/gpfs/zatkins/data/ACTCollaboration/tacos/examples/chi2_{type}_{chain.name}.png', bbox_inches='tight')

