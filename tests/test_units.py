import unittest
import numpy as np

from tacos import units

class TestUnits(unittest.TestCase):

    def test_db_dt(self):

        nu = 150e9
        nu_arr = np.asarray([90e9, 150e9])

        ans = units.db_dt(nu)
        ans_arr = units.db_dt(nu_arr)

        hplanck = 6.62607004e-34
        kboltz = 1.380649e-23
        clight  = 299792458 
        cmb_temp = 2.726
        
        x = hplanck * nu / kboltz / cmb_temp

        ans_exp = (2 * hplanck * nu ** 3 * x * np.exp(x) / cmb_temp / 
                   clight ** 2 / (np.exp(x) - 1) ** 2)

        x_arr = hplanck * nu_arr / kboltz / cmb_temp
        ans_arr_exp = (2 * hplanck * nu_arr ** 3 * x_arr * np.exp(x_arr) / cmb_temp / 
                   clight ** 2 / (np.exp(x_arr) - 1) ** 2)
        
        np.testing.assert_allclose(ans_arr / ans_arr_exp, np.ones(ans_arr_exp.size))

    def test_db_dt_divide_by_zero(self):

        nu = 0.
        ans = units.db_dt(nu)

        self.assertAlmostEqual(ans, 0.)

    def test_db_dt_overflow(self):

        # Large but but not crazy freq.
        nu = 1e12
        ans = units.db_dt(nu)
        ans_exp = 2.1518898299787773e-21
        self.assertAlmostEqual(ans, ans_exp)
        self.assertAlmostEqual(ans / ans_exp, 1)

        # Too large freq.
        nu = 1e20
        ans = units.db_dt(nu)
        ans_exp = 0.
        self.assertAlmostEqual(ans, ans_exp)

    def test_convert_rj_to_cmb(self):

        nu = np.linspace(1e9, 410e9, num=10000)
        bandpass = np.ones_like(nu)

        u_fact = units.convert_rj_to_cmb(bandpass, nu)
        u_fact_exp = 2.2348415733572202 # From Mathematica.

        self.assertAlmostEqual(u_fact, u_fact_exp, places=6)
