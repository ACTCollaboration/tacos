import unittest
import numpy as np

from tacos import bandpass, data

class TestBandPass(unittest.TestCase):

    def test_integrate_over_bandpass(self):

        nu = np.linspace(0, 10, num=100)
        signal = nu.copy()
        bp_arr = np.ones_like(nu)

        bp = bandpass.BandPass(bp_arr, nu)
        out = bp.integrate_over_bandpass(signal)

        # Bandpass will be normalized to have integral of 1.
        self.assertAlmostEqual(out, 5)

        signal_2d = np.ones((2, nu.size)) * nu
        out_2d = bp.integrate_over_bandpass(signal_2d)
        out_2d_exp = np.ones(2) * 5
        np.testing.assert_allclose(out_2d, out_2d_exp)

    def test_integrate_over_bandpass_axis(self):

        nu = np.linspace(0, 10, num=100)
        bp_arr = np.ones_like(nu)
        bp = bandpass.BandPass(bp_arr, nu)

        signal_3d = np.ones((3, nu.size, 2)) * nu[np.newaxis,:,np.newaxis]
        out_3d = bp.integrate_over_bandpass(signal_3d, axis=-2)
        out_3d_exp = np.ones((3, 2)) * 5

        np.testing.assert_allclose(out_3d, out_3d_exp)
                         
    def test_integrate_over_bandpass_data(self):

        instr_band = {}
        instr_band['act'] = ['f090', 'f150', 'f220']
        instr_band['planck'] = ['100', '143', '217', '353']
        instr_band['wmap'] = ['K', 'Ka', 'Q', 'V', 'W']

        # check each bandpass for normalization
        for instr in instr_band:
            for band in instr_band[instr]:
                band_int = data.Channel(instr=instr, band=band).bandpass.integrate_over_bandpass(np.ones(1))
                np.testing.assert_allclose(band_int, 1, atol=1e-6)
        

