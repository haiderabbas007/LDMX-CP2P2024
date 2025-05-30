"""
Unit tests for dark photon cross section analysis utilities.

This module tests the correctness and stability of core functions used in
the dark photon analysis script, including noisy data generation and
analytical cross section computation.

Tested Functions:
- generate_noisy_ratio_data
- compute_cross_section
"""

import unittest
import numpy as np
from your_script_filename import generate_noisy_ratio_data, compute_cross_section

class TestCrossSectionUtils(unittest.TestCase):
    """
    Unit test class for dark photon cross section utilities.
    """

    def test_generate_noisy_ratio_data_shape(self):
        """
        Test that output shapes from noisy data generator match input.
        """
        mass = np.linspace(0.001, 0.1, 50)
        coeffs = [1, -0.01, 0.0002, -1e-6]
        y_true, y_noisy = generate_noisy_ratio_data(mass, coeffs, noise_std=0.01)
        self.assertEqual(y_true.shape, mass.shape)
        self.assertEqual(y_noisy.shape, mass.shape)

    def test_generate_noisy_ratio_data_reproducibility(self):
        """
        Test reproducibility by fixing random seed.
        """
        np.random.seed(0)
        mass = np.array([0.01, 0.02])
        coeffs = [1, -0.01, 0.0002, -1e-6]
        _, y1 = generate_noisy_ratio_data(mass, coeffs, noise_std=0.01)

        np.random.seed(0)
        _, y2 = generate_noisy_ratio_data(mass, coeffs, noise_std=0.01)

        np.testing.assert_array_almost_equal(y1, y2, decimal=8)

    def test_compute_cross_section_positive_output(self):
        """
        Test that cross section computation returns positive result for physical inputs.
        """
        mA = 0.01  # GeV
        E0 = 4.0   # GeV
        Z = 74     # Tungsten
        A = 183.84
        sigma = compute_cross_section(mA, E0, Z, A)
        self.assertTrue(sigma > 0)

    def test_compute_cross_section_invalid_inputs(self):
        """
        Test that invalid inputs to compute_cross_section return 0.
        """
        Z, A = 74, 183.84
        self.assertEqual(compute_cross_section(0.0, 4.0, Z, A), 0.0)
        self.assertEqual(compute_cross_section(5.0, 4.0, Z, A), 0.0)  # mA > E0
        self.assertEqual(compute_cross_section(-1.0, 4.0, Z, A), 0.0)

if __name__ == '__main__':
    unittest.main()
