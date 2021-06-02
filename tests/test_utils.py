import numpy as np
from tacos import utils
def test_coadd_icovar():
    # make a dummy icov
    icov1 = np.array([[2,1,1], [1,2,1], [1,1,2]])[...,None,None]
    icov1 = np.tile(icov1, (1,1,10,10))
    icov2 = 2 * icov1

    icov = np.array([icov1, icov2])

    # make a dummy imap
    imap1 = np.full((3,10,10), 5)
    imap2 = np.full((3,10,10), 2)

    imap = np.array([imap1, imap2])

    # get coadd
    cmap, ccov = utils.get_coadd_map_covar(imap, icov, return_icovar_coadd=True)

    omap = np.full((3,10,10), 3)
    assert np.allclose(cmap, omap, rtol=0, atol=1e-14)

    ocov = np.array([[6,3,3],[3,6,3],[3,3,6]])[...,None,None]
    ocov = np.tile(ocov, (1,1,10,10))
    assert np.allclose(ccov, ocov, rtol=0, atol=1e-14)