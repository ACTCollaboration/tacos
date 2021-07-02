from tacos import utils, sampling, mixing_matrix
from tacos.utils import eigpow

from pixell import enmap, reproject
import pysm3

import matplotlib.pyplot as plt
import numpy as np
import argparse

from tqdm import tqdm

# parser = argparse.ArgumentParser('Generate an example chain of only amplitudes',
#                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('prior-icovar-factor', dest='prior_icovar_factor', type=float, help='Multiplicative factor applied to noise inverse-covariance to get prior inverse-covariance')
# parser.add_argument('--prior-offset', dest='prior_offset', type=float, default=100, help='Monopole [uK-RJ] to add to the pysm inputs, to represent prior mean')
# parser.add_argument('--num-steps', dest='num_steps', type=int, default=100, help='Number of Gibbs samples to draw')
# args = parser.parse_args()

prior_icovar_factor = 100#args.prior_icovar_factor
prior_offset = 100#args.prior_offset
num_steps = 100#args.num_steps

# first load our easy fixed quantities: data, noise covariance, prior mean, and mixing matrix 
config_path = 'examples/amplitudes_only.yaml'
name, channels, components, polstr, shape, wcs, kwargs = mixing_matrix._load_all_from_config(config_path, verbose=False)
odtype = kwargs.get('dtype', np.float32)

nchan = len(channels)
ncomp = len(components)
npol = len(polstr)
shape = shape[-2:]

d = enmap.enmap([c.map for c in channels], wcs=wcs, dtype=odtype)[:, 1:, ...]
N_inv = enmap.enmap([c.covmat for c in channels], wcs=wcs, dtype=odtype)[:, 1:, 1:, ...]
M = mixing_matrix.MixingMatrix.load_from_config(config_path)()

# next get our harder quantity: prior mean and prior covariance
sky = pysm3.Sky(nside=512, preset_strings=['s1', 'd1'], output_unit='uK_CMB')
pysm_s, pysm_d = sky.components

# get the amplitudes to project with our matrix from pysm
# include minus sign for IAU
a_s = np.array([pysm_s.I_ref.value, pysm_s.Q_ref.value, -pysm_s.U_ref.value])
a_d = np.array([pysm_d.I_ref.value, pysm_d.Q_ref.value, -pysm_d.U_ref.value])

# project amplitudes to CAR
a_s_car = reproject.enmap_from_healpix(a_s, shape, wcs, ncomp=3, rot=None)
a_d_car = reproject.enmap_from_healpix(a_d, shape, wcs, ncomp=3, rot=None)
a = np.array([a_s_car, a_d_car])[:, 1:, ...]
m = a + prior_offset

# first get projected N_inv and multiply by overall factor
MN_invM = np.einsum('jca...,jab...,jcb...->cab...', M, N_inv, M)
S_inv = prior_icovar_factor * MN_invM

# then, to make it non-trivial, let's double the diagonal
# diagonal goes to last axis, see np.diagonal
diagonal = np.diagonal(S_inv, axis1=1, axis2=2)

# this puts the diagonal into something of shape (ncomp, npol, npol, ...)
# so it can broadcast against arr
diagonal = np.einsum('ab,c...b->cab...', np.eye(npol, dtype=int), diagonal)
S_inv += diagonal

# let's make our various terms so the syntax is compact
F_inv = S_inv + MN_invM
F = eigpow(F_inv, -1, axes=(1, 2))

MN_invd = np.einsum('jca...,jab...,jb...->ca...', M, N_inv, d)
S_invm = np.einsum('cab...,cb...->ca...', S_inv, m)

c = np.einsum('cab...,cb...->ca...', F, MN_invd + S_invm)

N_halfinv = eigpow(N_inv, 0.5, axes=(1, 2))
MN_halfinv = np.einsum('jca...,jab...->jcab...', M, N_halfinv)

S_halfinv = eigpow(S_inv, 0.5, axes=(1, 2))

# let's now build our chain object, and define our chi^2 function
chain = sampling.Chain.load_from_config(config_path)

def chi2_per_pix(amp_samps):
    delta = amp_samps - c
    return np.einsum('...cayx,...cabyx,...cbyx->...yx', delta, F_inv, delta)

# now let's sample! this is pretty simple
for i in tqdm(range(num_steps)):
    np.random.seed(tuple((0, i)))
    eta_d = np.random.randn(*(nchan, npol, *shape))
    MN_halfinveta_d = np.einsum('jcab...,jb...->ca...', MN_halfinv, eta_d)

    np.random.seed(tuple((1, i)))
    eta_m = np.random.randn(*(ncomp, npol, *shape))
    S_halfinveta_s = np.einsum('cab...,cb...->ca...', S_halfinv, eta_m)

    # solve for our sample
    RHS = MN_invd + S_invm + MN_halfinveta_d + S_halfinveta_s
    x = np.einsum('cab...,cb...->ca...', F, RHS)
    weight = [1, chi2_per_pix(x).mean()]

    # update the chain
    chain.append_weights(weight)
    chain.append_amplitudes(x)