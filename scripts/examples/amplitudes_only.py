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

# get whether this is prior or data dominated by "name"
config_path = args.config_path
prior = config_path.split('_dom')[0].split('only_')[1] == 'prior'
print(f'Prior_dom: {prior}')

# get run paramaters
name, channels, components, polstr, shape, wcs, kwargs = mixing_matrix._load_all_from_config(config_path, verbose=False)
odtype = kwargs.get('dtype', np.float32)

nchan = len(channels)
ncomp = len(components)
npol = len(polstr)
shape = shape[-2:]
polidxs = np.array(['IQU'.index(char) for char in polstr])

# get run arguments
config = utils.config_from_yaml_resource(config_path)
prior_icovar_factor = config['parameters']['prior_icovar_factor']
prior_offset = config['parameters']['prior_offset']
num_steps = config['parameters']['num_steps']

# first load our easy fixed quantities: data, noise covariance, and mixing matrix 
d = enmap.enmap([c.map for c in channels], wcs=wcs, dtype=odtype)[:, polidxs, ...]
N_inv = enmap.enmap([c.covmat for c in channels], wcs=wcs, dtype=odtype)[(slice(None), *np.ix_(polidxs, polidxs))]
M = mixing_matrix.MixingMatrix.load_from_config(config_path)()

# next get our harder quantity: prior mean and prior covariance
sky = pysm3.Sky(nside=512, preset_strings=['d1', 's1'], output_unit='uK_CMB')
pysm_d, pysm_s = sky.components

# get the amplitudes to project with our matrix from pysm
# include minus sign for IAU
a_d = np.array([pysm_d.I_ref.value, pysm_d.Q_ref.value, -pysm_d.U_ref.value])
a_s = np.array([pysm_s.I_ref.value, pysm_s.Q_ref.value, -pysm_s.U_ref.value])

# project amplitudes to CAR
a_d_car = reproject.enmap_from_healpix(a_d, shape, wcs, ncomp=3, rot=None)
a_s_car = reproject.enmap_from_healpix(a_s, shape, wcs, ncomp=3, rot=None)
a = np.array([a_d_car, a_s_car])[:, polidxs, ...]
m = a + prior_offset

# first get projected N_inv and multiply by overall factor
MN_invM = np.einsum('jca...,jab...,jdb...->cadb...', M, N_inv, M)
S_inv = prior_icovar_factor * MN_invM

# then, to make it non-trivial, let's double the diagonal
# first flatten the correlated axes and then unflatten afterwards
diagonal = S_inv.reshape(ncomp*npol, ncomp*npol, *S_inv.shape[-2:])

# # diagonal goes to last axis, see np.diagonal
diagonal = np.diagonal(diagonal, axis1=0, axis2=1)

# this puts the diagonal back into something of shape (ncomp*npol, ncomp*npol, ...)
# so it can broadcast against arr
diagonal = np.einsum('ab,...b->ab...', np.eye(ncomp*npol, dtype=int), diagonal)
S_inv += diagonal.reshape(ncomp, npol, ncomp, npol, *S_inv.shape[-2:])

# let's make our various terms so the syntax is compact
F_inv = S_inv + MN_invM
F = eigpow(F_inv.reshape(ncomp*npol, ncomp*npol, *F_inv.shape[-2:]), -1, axes=(0, 1))
F = F.reshape(ncomp, npol, ncomp, npol, *F.shape[-2:])

MN_invd = np.einsum('jca...,jab...,jb...->ca...', M, N_inv, d)

S_invm = np.einsum('cadb...,ca...->db...', S_inv, m)

c = np.einsum('cadb...,ca...->db...', F, MN_invd + S_invm)

N_halfinv = eigpow(N_inv, 0.5, axes=(1, 2))
MN_halfinv = np.einsum('jca...,jab...->jcab...', M, N_halfinv)

S_halfinv = eigpow(S_inv.reshape(ncomp*npol, ncomp*npol, *S_inv.shape[-2:]), 0.5, axes=(0, 1))
S_halfinv = S_halfinv.reshape(ncomp, npol, ncomp, npol, *S_halfinv.shape[-2:])

# let's now build our chain object, and define our chi^2 function
chain = chain.Chain.load_from_config(config_path)

def chi2_per_pix(amp_samps, mean=c):
    delta = amp_samps - mean
    return np.einsum('...cayx,...cadbyx,...dbyx->...yx', delta, F_inv, delta)

# now let's sample! this is pretty simple
for i in tqdm(range(num_steps)):
    np.random.seed(tuple((0, i)))
    eta_d = np.random.randn(*(nchan, npol, *shape), dtype=np.float32)
    MN_halfinveta_d = np.einsum('jcab...,jb...->ca...', MN_halfinv, eta_d)

    np.random.seed(tuple((1, i)))
    eta_m = np.random.randn(*(ncomp, npol, *shape), dtype=np.float32)
    S_halfinveta_s = np.einsum('cadb...,ca...->db...', S_halfinv, eta_m)

    # solve for our sample
    RHS = MN_invd + S_invm + MN_halfinveta_d + S_halfinveta_s
    x = np.einsum('cadb...,ca...->db...', F, RHS)
    weight = [1, chi2_per_pix(x).mean()]

    # update the chain
    chain.add_samples(weights=weight, amplitudes=x)

chain.write_samples(overwrite=True)