"""
This module contains unit tests for the `generate_noisy_ratio_data` function
defined in `random_numbers.py`. These tests verify the shape, statistical properties,
and deterministic behavior (with a seed) of the noisy data generator.

These tests are essential for ensuring the reliability of synthetic data used in
Dark Photon cross section analysis workflows.
"""

import unittest
import numpy as np
from random_numbers import generate_noisy_ratio_data

class TestGenerateNoisyRatioData(unittest.TestCase):
    """
    Unit tests for the generate_noisy_ratio_data function in random_numbers.py.
    """

    def setUp(self):
        """
        Prepare test data including mass points and polynomial coefficients.
        """
        self.mass_points = np.linspace(0.001, 0.1, 50)
        self.coeffs = [1.0, -0.01, 2e-4, -1e-6]
        self.noise_std = 0.05

    def test_output_shape_matches_input(self):
        """
        Test that the output array has the same shape as the input mass points.
        """
        y_noisy = generate_noisy_ratio_data(self.mass_points, self.coeffs, self.noise_std)
        self.assertEqual(y_noisy.shape, self.mass_points.shape)

    def test_output_type_is_numpy_array(self):
        """
        Test that the output is a numpy ndarray.
        """
        y_noisy = generate_noisy_ratio_data(self.mass_points, self.coeffs, self.noise_std)
        self.assertIsInstance(y_noisy, np.ndarray)

    def test_deterministic_output_with_seed(self):
        """
        Test that the output is deterministic if the same random seed is set.
        """
        np.random.seed(42)
        y1 = generate_noisy_ratio_data(self.mass_points, self.coeffs, self.noise_std)

        np.random.seed(42)
        y2 = generate_noisy_ratio_data(self.mass_points, self.coeffs, self.noise_std)

        np.testing.assert_array_almost_equal(y1, y2)

    def test_noise_std_dev_approximate(self):
        """
        Test that the standard deviation of noise is close to the specified value.
        """
        np.random.seed(123)
        y_noisy = generate_noisy_ratio_data(self.mass_points, self.coeffs, self.noise_std)

        x = self.mass_points * 1000
        y_true = self.coeffs[0] + self.coeffs[1]*x + self.coeffs[2]*x**2 + self.coeffs[3]*x**3
        estimated_std = np.std(y_noisy - y_true)

        self.assertAlmostEqual(estimated_std, self.noise_std, delta=0.02)

if __name__ == "__main__":
    unittest.main()
